/**
*	Based off earlier start from:
*	https://github.com/Robadob/SP-Bench/commit/35dcbb81cc0b73cdb6b08fb622f13e688a878133
*	This technique only concerns 2D FoV limitation
*	In particular as may be applied to pedestrian models, whereby a pedestrian is assumed unable to see behind them.
*	We attempt to utilise a look-up table (similar to marching cubes), to optimise calculation of bins which intersect the given FoV
*/
#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/constants.hpp>
#include <curand_kernel.h>
#include <texture_fetch_functions.h>
#include <cub/cub.cuh>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/vector_angle.hpp>
#define EPSILON 0.0001f
#define DEBUG_INDEX 0
//#define CIRCLES
//Cuda call
static void HandleCUDAError(const char *file,
	int line,
	cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
	cudaDeviceSynchronize();
#endif
	if (status != cudaError::cudaSuccess || (status = cudaGetLastError()) != cudaError::cudaSuccess)
	{
		printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line, cudaGetErrorString(status));
#ifdef _DEBUG
		getchar();
#endif
		exit(1);
	}
}
#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))
#define CUDA_CHECK() (HandleCUDAError(__FILE__, __LINE__))

//Logging (found in log.cpp)
#include <fstream>
void createLog(std::ofstream &f);
void log(std::ofstream &f,
	const unsigned int &estRadialNeighbours,
	const unsigned int &agentCount,
	const unsigned int &envWidth,
	const float &PBM_control,
	const float &kernel_control,
	const float &PBM,
	const float &kernel,
	const unsigned int &fails
);
__device__ __constant__ unsigned int d_agentCount;
__device__ __constant__ float d_environmentWidth_float;
__device__ __constant__ unsigned int d_gridDim;
glm::uvec2 GRID_DIMS;
__device__ __constant__ float d_gridDim_float;
__device__ __constant__ float d_RADIUS;
__device__ __constant__ float d_R_SIN_45;
__device__ __constant__ float d_binWidth;

texture<float4> d_texMessages;//float2 pos, float2 velocity
texture<unsigned int> d_texPBM;

__global__ void init_curand(curandState *state, unsigned long long seed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < d_agentCount)
		curand_init(seed, id, 0, &state[id]);
}
__global__ void init_agents(curandState *state, glm::vec4 *locationMessages) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= d_agentCount)
		return;
	//Position
	//curand_unform returns 0<x<=1.0, not much can really do about 0 exclusive
	//negate and  + 1.0, to make  0<=x<1.0
	locationMessages[id].x = (-curand_uniform(&state[id]) + 1.0f)*d_environmentWidth_float;
	locationMessages[id].y = (-curand_uniform(&state[id]) + 1.0f)*d_environmentWidth_float;
	//Velocity
	glm::vec2 vel = normalize(glm::vec2((curand_uniform(&state[id])-0.5)*2.0f, (curand_uniform(&state[id])-0.5f)*2.0f));
	locationMessages[id].z = vel.x;
	locationMessages[id].w = vel.y;
}
__device__ __forceinline__ glm::ivec2 getGridPosition(glm::vec2 worldPos)
{
	//Clamp each grid coord to 0<=x<dim
	return clamp(floor((worldPos / d_environmentWidth_float)*d_gridDim_float), glm::vec2(0), glm::vec2((float)d_gridDim - 1));
}
__device__ __forceinline__ glm::ivec2 _getGridPosition(glm::vec2 worldPos)
{
	return floor((worldPos / d_environmentWidth_float)*d_gridDim_float);
}
__device__ __forceinline__ unsigned int getHash(glm::ivec2 gridPos)
{
	//Bound gridPos to gridDimensions
	gridPos = clamp(gridPos, glm::ivec2(0), glm::ivec2(d_gridDim - 1));
	//Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
	return (unsigned int)(
		(gridPos.y * d_gridDim) +					//y
		gridPos.x); 	                            //x
}
__global__ void atomicHistogram(unsigned int* bin_index, unsigned int* bin_sub_index, unsigned int *pbm_counts, glm::vec4 *messageBuffer)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;

	glm::ivec2 gridPos = getGridPosition(glm::vec2(messageBuffer[index].x, messageBuffer[index].y));
	unsigned int hash = getHash(gridPos);
	bin_index[index] = hash;
	unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
	bin_sub_index[index] = bin_idx;
}
__global__ void reorderLocationMessages(
	unsigned int* bin_index,
	unsigned int* bin_sub_index,
	unsigned int *pbm,
	glm::vec4 *unordered_messages,
	glm::vec4 *ordered_messages
)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;

	unsigned int i = bin_index[index];
	unsigned int sorted_index = pbm[i] + bin_sub_index[index];

	//Order messages into swap space
	ordered_messages[sorted_index] = unordered_messages[index];
}
__forceinline__ __device__ void avoidSum(const glm::vec2 &mePos, const glm::vec2 &meVec, const glm::vec2 &msgPos, const glm::vec2 &msgVec, glm::vec2 &nVel, glm::vec2 &aVel)
{
#define SPEED_LIMIT 1.0f
#define TIME_SCALER	0.0003f
#define MIN_DISTANCE 0.0001f
#define SCALE_FACTOR 0.03125
#define I_SCALER (SCALE_FACTOR*0.35f)
#define STEER_WEIGHT		0.10f
#define AVOID_WEIGHT		0.02f
#define COLLISION_WEIGHT	0.50f
#define GOAL_WEIGHT			0.20f
	//Lightweight bounds check
	glm::vec2 offset = msgPos - mePos;
	float distance = glm::length(offset);
	if (distance <d_RADIUS && distance > MIN_DISTANCE)
	{
		//FOV Check
		float angle = glm::angle(glm::normalize(meVec), glm::normalize(offset));
		if (angle<1.5708)//d_HALF_FOV (90 degrees in radians)
		{
			float perception = 45.0f;
			//STEER
			if ((angle < glm::radians(perception)) || (angle > 3.14159265f - glm::radians(perception))) {
				glm::vec2 s_velocity = -offset;
				s_velocity *= powf(I_SCALER / distance, 1.25f)*STEER_WEIGHT;
				nVel += s_velocity;
			}

			//AVOID
			glm::vec2 a_velocity = -offset;
			a_velocity *= powf(I_SCALER / distance, 2.00f)*AVOID_WEIGHT;
			aVel += a_velocity;
		}
	}

}
/**
* Kernel must be launched 1 block per bin
* This removes the necessity of __launch_bounds__(64) as all threads in block are touching the same messages
* However we end up with alot of (mostly) idle threads if one bin dense, others empty.
*/
__global__  void __launch_bounds__(64) neighbourSearch_control(const glm::vec4 *agents, glm::vec4 *out)
{
//#define STRIPS
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;
	glm::vec2 pos = glm::vec2(agents[index].x, agents[index].y);
	glm::vec2 vel = glm::vec2(agents[index].z, agents[index].w);
	glm::vec2 navigate_velocity = glm::vec2(0);
	glm::vec2 avoid_velocity = glm::vec2(0);
	glm::ivec2 gridPos = getGridPosition(pos);
	glm::ivec2 gridPosRelative;

	for (gridPosRelative.y = -1; gridPosRelative.y <= 1; gridPosRelative.y++)
	//for (gridPosRelative.y = 0; gridPosRelative.y <= 0; gridPosRelative.y++)
	{//ymin to ymax
		int currentBinY = gridPos.y + gridPosRelative.y;
		if (currentBinY >= 0 && currentBinY < d_gridDim)
		{
#ifndef STRIPS
			for (gridPosRelative.x = -1; gridPosRelative.x <= 1; gridPosRelative.x++)
			//for (gridPosRelative.x = 0; gridPosRelative.x <= 0; gridPosRelative.x++)
			{//xmin to xmax
				int currentBinX = gridPos.x + gridPosRelative.x;
				if (currentBinX >= 0 && currentBinX < d_gridDim)
				{
					//Find bin start and end
					unsigned int binHash = getHash(glm::ivec2(currentBinX, currentBinY));
					unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
					unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
#ifdef _DEBUG
					if (index == DEBUG_INDEX)
						printf("bin: (%d, %d)[%d, %d]\n", currentBinX, currentBinY, gridPosRelative.x, gridPosRelative.y);
#endif
#else

					int currentBinX = gridPos.x - 1;
					currentBinX = currentBinX >= 0 ? currentBinX : 0;
					unsigned int binHash = getHash(glm::ivec2(currentBinX, currentBinY));
					unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
					currentBinX = gridPos.x + 1;
					currentBinX = currentBinX < d_gridDim ? currentBinX : d_gridDim - 1;
					binHash = getHash(glm::ivec2(currentBinX, currentBinY));
					unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
#endif
					//Iterate messages in range
					for (unsigned int i = binStart; i < binEnd; ++i)
					{
						if (i != index)
						{
							float4 message = tex1Dfetch(d_texMessages, i);
							glm::vec2 *_pos = (glm::vec2*)&message;
							glm::vec2 *_vel = (glm::vec2*)&(message.z);
#ifdef _DEBUG
							if (index == DEBUG_INDEX)
							{
								glm::vec2 offset = *_pos - pos;
								float distance = glm::length(offset);
								//FOV Check
								float angle = glm::angle(glm::normalize(vel), glm::normalize(offset));
								if (angle < 1.5708 && distance <d_RADIUS && distance > MIN_DISTANCE)//d_HALF_FOV (90 degrees in radians)
								{
									printf("%d\n", i);
								}
								else
									printf("-%d\n", i);
							}
#endif
							avoidSum(pos, vel, *_pos, *_vel, navigate_velocity, avoid_velocity);
						}
					}
#ifndef STRIPS
				}
			}
#endif
		}
	}

	//Process result of avoidsum
	{
		//random walk goal
		glm::vec2 goal_velocity = vel * GOAL_WEIGHT;

		//maximum velocity rule
		goal_velocity += navigate_velocity + avoid_velocity;

		float current_speed = length(vel) + 0.025f;
		vel += current_speed * goal_velocity;
		float speed = length(vel);
		//limit speed
		if (speed >= SPEED_LIMIT) {
			vel = normalize(vel)*SPEED_LIMIT;
			speed = SPEED_LIMIT;
		}

		//update position
		pos += vel*TIME_SCALER;
	}
#ifdef _DEBUG
	if (index == DEBUG_INDEX)
		printf("pos(%.5f, %.5f), vel(%.5f, %.5f)\n", pos.x, pos.y, vel.x, vel.y);
#endif
	out[index] = glm::vec4(pos, vel);
}
/**
* Kernel must be launched 1 block per bin
* This removes the necessity of __launch_bounds__(64) as all threads in block are touching the same messages
* However we end up with alot of (mostly) idle threads if one bin dense, others empty.
*/
__global__ void __launch_bounds__(64) neighbourSearch(const glm::vec4 *agents, glm::vec4 *out)
{
	const glm::ivec2 relatives[8] = { 
		glm::ivec2(0, 1),	//North
		glm::ivec2(1,1),	//North East
		glm::ivec2(1,0),    //East
		glm::ivec2(1, -1),	//South East
		glm::ivec2(0, -1),	//South
		glm::ivec2(-1, -1), //South West
		glm::ivec2(-1, 0),	//West
		glm::ivec2(-1, 1)	//North West
	};

	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;

	//My data
	glm::vec2 pos = glm::vec2(agents[index].x, agents[index].y);
	glm::vec2 vel = glm::vec2(agents[index].z, agents[index].w);
	glm::vec2 navigate_velocity = glm::vec2(0);
	glm::vec2 avoid_velocity = glm::vec2(0);
	glm::ivec2 myBin = getGridPosition(pos);
	{
		//Process relative (0, 0)
		unsigned int binHash = getHash(myBin);
		unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
		unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
#ifdef _DEBUG
		if (index == DEBUG_INDEX)
			printf("bin: (%d, %d)[%d, %d]\n", myBin.x, myBin.y, 0, 0);
#endif
		for (unsigned int j = binStart; j<binEnd; ++j)
		{
			if (j != index)
			{
				float4 message = tex1Dfetch(d_texMessages, j);
				glm::vec2 *_pos = (glm::vec2*)&message;
				glm::vec2 *_vel = (glm::vec2*)&(message.z);
#ifdef _DEBUG
				if (index == DEBUG_INDEX)
				{
					glm::vec2 offset = *_pos - pos;
					float distance = glm::length(offset);
					//FOV Check
					float angle = glm::angle(glm::normalize(vel), glm::normalize(offset));
					if (angle < 1.5708 && distance <d_RADIUS && distance > MIN_DISTANCE)//d_HALF_FOV (90 degrees in radians)
					{
						printf("%d\n", j);
					}
					else
						printf("-%d\n", j);
				}
#endif
				avoidSum(pos, vel, *_pos, *_vel, navigate_velocity, avoid_velocity);
			}
		}
	}
#define METHOD2
#ifndef METHOD2
	//Identify the relative element which contains dir
	unsigned int __relativeIndex;
	unsigned int __relativeCount;
	{
		//This technique takes central point and estimates sides
		////incremenet pos by vel * unit
		//glm::vec2 _dest = pos - glm::vec2(myBin)*d_binWidth;
		//glm::vec2 t;
		////Identify distance to next cell according to velocity
		//t.x = vel.x > 0 ? 1 - _dest.x : -(_dest.x);
		//t.y = vel.y > 0 ? 1 - _dest.y : -(_dest.y);
		////Convert this distance into time
		//t.x = vel.x != 0 ? t.x / vel.x : FLT_MAX;
		//t.y = vel.y != 0 ? t.y / vel.y : FLT_MAX;
		////Extend pos to next cell
		//glm::ivec2 relativeBin;
		//relativeBin.x = t.x <= t.y ? t.x / abs(t.x) : 0;
		//relativeBin.y = t.y <= t.x ? t.y / abs(t.y) : 0;
		//glm::ivec2 destOffset = myBin + relativeBin;
		//assert(destOffset != glm::ivec2(0));
		////Identify index where that falls in 'relatives' array
		//if (destOffset.x == 1)
		//{
		//	__relativeIndex = 2 - destOffset.y;
		//}
		//else if (destOffset.x == -1)
		//{
		//	__relativeIndex = 6 - destOffset.y;
		//}
		//else
		//{
		//	__relativeIndex = 2 - 2 * destOffset.y;
		//}
		////Rotate about circle -FOV/2 (how many elements is this?
		////180 degrees requires 2 on either side of central
		//__relativeIndex -= 2;
		//__relativeCount = 5;
		////
		//glm::vec2 qPos = pos - glm::vec2(glm::ivec2(pos));//Just want the decimal part
		//if (qPos.x > 0)qPos.x = 1;
		//else if (qPos.x < 0)qPos.x = -1;
		//if (qPos.y > 0)qPos.y = 1;
		//else if (qPos.y < 0)qPos.y = -1;
		//glm::ivec2 _qPos = qPos;
		////+-1 on either side, based on the quadrant relative to velocity
		////Temp(?) max all
		//__relativeIndex -= 1;
		//__relativeCount += 2;
		////Correct for overflow
		//__relativeIndex += 64;// +8*8 to account for underflow (remainder op inside for loop handles overflow)
	}
#else
	//Identify the relative element for left FOV limit
	//incremenet pos by vel * unit
	unsigned int __relativeIndex_left;
	unsigned int __relativeIndex_right;
	unsigned int __furthestIndex_left;
	unsigned int __furthestIndex_right;
	const glm::vec2 _dest = pos - glm::vec2(myBin)*d_binWidth;
	{
		glm::vec2 left = glm::vec2(-vel.y, vel.x);
		{//First intersection
			glm::ivec2 relativeBin;
			glm::vec2 t;
			//Identify distance to next cell according to velocity
			t.x = left.x > 0 ? 1 - _dest.x : -(_dest.x);
			t.y = left.y > 0 ? 1 - _dest.y : -(_dest.y);
			//Convert this distance into time
			t.x = left.x != 0 ? t.x / left.x : FLT_MAX;
			t.y = left.y != 0 ? t.y / left.y : FLT_MAX;
			//Extend pos to next cell
			relativeBin.x = t.x <= t.y ? left.x / abs(left.x) : 0;
			relativeBin.y = t.y <= t.x ? left.y / abs(left.y) : 0;
			assert(relativeBin != glm::ivec2(0));
			//Identify index where that falls in 'relatives' array
			if (relativeBin.x == 1)
			{
				__relativeIndex_left = 2 - relativeBin.y;
			}
			else if (relativeBin.x == -1)
			{
				__relativeIndex_left = 6 - relativeBin.y;
			}
			else
			{
				__relativeIndex_left = 2 - 2 * relativeBin.y;
			}
		}
		{//Last intersection
		 //Identify distance to edge of radius
			glm::ivec2 furthestBin = getGridPosition(pos + d_RADIUS * glm::normalize(left)) - myBin;
			//Identify index where that falls in 'relatives' array
			if (furthestBin.x == 1)
			{
				__furthestIndex_left = 2 - furthestBin.y;
			}
			else if (furthestBin.x == -1)
			{
				__furthestIndex_left = 6 - furthestBin.y;
			}
			else
			{
				__furthestIndex_left = 2 - 2 * furthestBin.y;
			}
		}
	}
	{
		glm::vec2 right = glm::vec2(vel.y, -vel.x);
		{//First intersection
			glm::ivec2 relativeBin;
			glm::vec2 t;
			//Identify distance to next cell according to velocity
			t.x = right.x > 0 ? 1 - _dest.x : -(_dest.x);
			t.y = right.y > 0 ? 1 - _dest.y : -(_dest.y);
			//Convert this distance into time
			t.x = right.x != 0 ? t.x / right.x : FLT_MAX;
			t.y = right.y != 0 ? t.y / right.y : FLT_MAX;
			//Extend pos to next cell
			relativeBin.x = t.x <= t.y ? right.x / abs(right.x) : 0;
			relativeBin.y = t.y <= t.x ? right.y / abs(right.y) : 0;
			assert(relativeBin != glm::ivec2(0));
			//Identify index where that falls in 'relatives' array
			if (relativeBin.x == 1)
			{
				__relativeIndex_right = 2 - relativeBin.y;
			}
			else if (relativeBin.x == -1)
			{
				__relativeIndex_right = 6 - relativeBin.y;
			}
			else
			{
				__relativeIndex_right = 2 - 2 * relativeBin.y;
			}
		}
		{//Last intersection
		 //Identify distance to edge of radius
			glm::ivec2 furthestBin = getGridPosition(pos + d_RADIUS * glm::normalize(right)) - myBin;
			//Identify index where that falls in 'relatives' array
			if (furthestBin.x == 1)
			{
				__furthestIndex_right = 2 - furthestBin.y;
			}
			else if (furthestBin.x == -1)
			{
				__furthestIndex_right = 6 - furthestBin.y;
			}
			else
			{
				__furthestIndex_right = 2 - 2 * furthestBin.y;
			}
		}
	}
	unsigned int __relativeCount;
	{//Detect which indexes creates the widest range
		{//left
			if(__relativeIndex_left != __furthestIndex_left)
			{
				unsigned int _relativeCount = __relativeIndex_right - __relativeIndex_left + 9;
				_relativeCount = _relativeCount <= 8 ? _relativeCount : _relativeCount - 8;
				unsigned int __furthestCount = __relativeIndex_right - __furthestIndex_left + 9;
				__furthestCount = __furthestCount <= 8 ? __furthestCount : __furthestCount - 8;
				__relativeIndex_left = _relativeCount > __furthestCount ? __relativeIndex_left : __furthestIndex_left;
			}
		}
		{//right
			//if (__relativeIndex_right != __furthestIndex_right)
			{
				unsigned int _relativeCount = __relativeIndex_right - __relativeIndex_left + 9;
				_relativeCount = _relativeCount <= 8 ? _relativeCount : _relativeCount - 8;
				unsigned int __furthestCount = __furthestIndex_right - __relativeIndex_left + 9;
				__furthestCount = __furthestCount <= 8 ? __furthestCount : __furthestCount - 8;
				__relativeCount = _relativeCount > __furthestCount ? _relativeCount : __furthestCount;
				__relativeIndex_right = _relativeCount > __furthestCount ? __relativeIndex_right : __furthestIndex_right;
			}
		}
	}
#endif
	//Iterate FOV relatives across
#ifndef METHOD2
	for(unsigned int i = 0;i<__relativeCount;++i)
	{
		int currentIndex = __relativeIndex + i;
		currentIndex = currentIndex % 8;
#else
	assert( __relativeCount < 9);
	for (unsigned int i = 0; i<__relativeCount; ++i)
	{
		unsigned int currentIndex = __relativeIndex_left + i;
		currentIndex = currentIndex < 8 ? currentIndex : currentIndex - 8;
		assert(currentIndex < 8);
#endif
		glm::ivec2 currentBin = myBin + relatives[currentIndex];
		if (currentBin.x >= 0 && currentBin.x < d_gridDim)
		{
			if (currentBin.y >= 0 && currentBin.y < d_gridDim)
			{
				//Now we must load all messages from currentBin
				unsigned int binHash = getHash(currentBin);
				unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
				unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
#ifdef _DEBUG
				if (index == DEBUG_INDEX)
					printf("bin: (%d, %d)[%d, %d]\n", currentBin.x, currentBin.y, relatives[currentIndex].x, relatives[currentIndex].y);
#endif
				for(unsigned int j = binStart;j<binEnd;++j)
				{
					float4 message = tex1Dfetch(d_texMessages, j);
					glm::vec2 *_pos = (glm::vec2*)&message;
					glm::vec2 *_vel = (glm::vec2*)&(message.z);
#ifdef _DEBUG
					if (index == DEBUG_INDEX)
					{
						glm::vec2 offset = *_pos - pos;
						float distance = glm::length(offset);
						//FOV Check
						float angle = glm::angle(glm::normalize(vel), glm::normalize(offset));
						if (angle < 1.5708 && distance <d_RADIUS && distance > MIN_DISTANCE)//d_HALF_FOV (90 degrees in radians)
						{
							printf("%d\n", j);
						}
						else
							printf("-%d\n", j);
					}
#endif
					avoidSum(pos, vel, *_pos, *_vel, navigate_velocity, avoid_velocity);
				}
			}
		}
				
	}

	//Process result of avoid sum
	{
		//random walk goal
		glm::vec2 goal_velocity = vel * GOAL_WEIGHT;

		//maximum velocity rule
		goal_velocity += navigate_velocity + avoid_velocity;

		float current_speed = length(vel) + 0.025f;
		vel += current_speed * goal_velocity;
		float speed = length(vel);
		//limit speed
		if (speed >= SPEED_LIMIT) {
			vel = normalize(vel)*SPEED_LIMIT;
			speed = SPEED_LIMIT;
		}

		//update position
		pos += vel*TIME_SCALER;
	}
#ifdef _DEBUG
	if (index == DEBUG_INDEX)
		printf("pos(%.5f, %.5f), vel(%.5f, %.5f)\n", pos.x, pos.y, vel.x, vel.y);
#endif
	//Output
	out[index] = glm::vec4(pos, vel);
}


__global__ void unsortMessages(
	unsigned int* bin_index,
	unsigned int* bin_sub_index,
	unsigned int *pbm,
	glm::vec4 *ordered_messages,
	glm::vec4 *unordered_messages
)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;

	unsigned int i = bin_index[index];
	unsigned int sorted_index = pbm[i] + bin_sub_index[index];

	//Order messages into swap space
	unordered_messages[index] = ordered_messages[sorted_index];
}
/**
* This program is to act as a test rig to demonstrate the raw impact of raw message handling
*/
void run(std::ofstream &f, const unsigned int ENV_WIDTH, const unsigned int AGENT_COUNT = 1000000)
{
	void *d_CUB_temp_storage = nullptr;
	size_t d_CUB_temp_storage_bytes = 0;
	//Spatial partitioning mock
	//Fixed 2D environment of 1000x1000
	//Filled with 1,000,000 randomly distributed agents
	//const unsigned int ENV_WIDTH = 250;
	float ENV_WIDTH_float = (float)ENV_WIDTH;
	const unsigned int RNG_SEED = 12;
	const unsigned int ENV_VOLUME = ENV_WIDTH * ENV_WIDTH;
	CUDA_CALL(cudaMemcpyToSymbol(d_agentCount, &AGENT_COUNT, sizeof(unsigned int)));
	printf("Env Width: %g\n", ENV_WIDTH_float);
	CUDA_CALL(cudaMemcpyToSymbol(d_environmentWidth_float, &ENV_WIDTH_float, sizeof(float)));
	glm::vec4 *d_agents_init = nullptr, *d_agents = nullptr, *d_out = nullptr;
	unsigned int *d_keys = nullptr, *d_vals = nullptr;
	CUDA_CALL(cudaMalloc(&d_agents_init, sizeof(glm::vec4) * AGENT_COUNT));
	CUDA_CALL(cudaMalloc(&d_agents, sizeof(glm::vec4) * AGENT_COUNT));
	CUDA_CALL(cudaMalloc(&d_out, sizeof(glm::vec4) * AGENT_COUNT));
	glm::vec4 *h_out = (glm::vec4*)malloc(sizeof(glm::vec4) * AGENT_COUNT);
	glm::vec4 *h_out_control = (glm::vec4*)malloc(sizeof(glm::vec4) * AGENT_COUNT);
	//Init agents
	{
		//Generate curand
		curandState *d_rng;
		CUDA_CALL(cudaMalloc(&d_rng, AGENT_COUNT * sizeof(curandState)));
		//Arbitrary thread block sizes (speed not too important during one off initialisation)
		unsigned int initThreads = 512;
		unsigned int initBlocks = (AGENT_COUNT / initThreads) + 1;
		init_curand << <initBlocks, initThreads >> >(d_rng, RNG_SEED);//Defined in CircleKernels.cuh
		CUDA_CALL(cudaDeviceSynchronize());
		init_agents << <initBlocks, initThreads >> >(d_rng, d_agents_init);
		//Free curand
		CUDA_CALL(cudaFree(d_rng));
		CUDA_CALL(cudaMalloc(&d_keys, sizeof(unsigned int)*AGENT_COUNT));
		CUDA_CALL(cudaMalloc(&d_vals, sizeof(unsigned int)*AGENT_COUNT));
	}
	//Decide interaction radius
	//for a range of bin widths
	const float RADIUS = 1.0f;//
	const float RADIAL_VOLUME = glm::pi<float>()*RADIUS*RADIUS;
	const unsigned int AVERAGE_NEIGHBOURS = (unsigned int)(AGENT_COUNT*RADIAL_VOLUME / ENV_VOLUME);
	printf("Agents: %d, RVol: %.2f, Average Neighbours: %d\n", AGENT_COUNT, RADIAL_VOLUME, AVERAGE_NEIGHBOURS);
	//{
	//    cudaFree(d_agents_init);
	//    cudaFree(d_agents);
	//    cudaFree(d_out);
	//    return;
	//}

	const float rSin45 = (float)(RADIUS*sin(glm::radians(45)));
	CUDA_CALL(cudaMemcpyToSymbol(d_RADIUS, &RADIUS, sizeof(float)));
	CUDA_CALL(cudaMemcpyToSymbol(d_R_SIN_45, &rSin45, sizeof(float)));
	{
		{
			//Copy init state to d_out   
			CUDA_CALL(cudaMemcpy(d_out, d_agents_init, sizeof(glm::vec4)*AGENT_COUNT, cudaMemcpyDeviceToDevice));
		}
		//Decide bin width (as a ratio to radius)
		const float BIN_WIDTH = RADIUS;
		float GRID_DIMS_float = ENV_WIDTH / BIN_WIDTH;
		GRID_DIMS = glm::uvec2((unsigned int)ceil(GRID_DIMS_float));
		CUDA_CALL(cudaMemcpyToSymbol(d_binWidth, &BIN_WIDTH, sizeof(float)));
		CUDA_CALL(cudaMemcpyToSymbol(d_gridDim, &GRID_DIMS.x, sizeof(unsigned int)));
		CUDA_CALL(cudaMemcpyToSymbol(d_gridDim_float, &GRID_DIMS_float, sizeof(float)));
		printf("Grid Dims: %g\n", GRID_DIMS_float);
		printf("Bin Width: %g\n", BIN_WIDTH);
		const unsigned int BIN_COUNT = glm::compMul(GRID_DIMS);
		cudaEvent_t start_PBM, end_PBM, start_kernel, end_kernel;
		cudaEventCreate(&start_PBM);
		cudaEventCreate(&end_PBM);
		cudaEventCreate(&start_kernel);
		cudaEventCreate(&end_kernel);
		//BuildPBM
		unsigned int *d_PBM_counts = nullptr;
		unsigned int *d_PBM = nullptr;
		CUDA_CALL(cudaMalloc(&d_PBM_counts, (BIN_COUNT + 1) * sizeof(unsigned int)));
		CUDA_CALL(cudaMalloc(&d_PBM, (BIN_COUNT + 1) * sizeof(unsigned int)));
		{//Resize cub temp if required
			size_t bytesCheck;
			cub::DeviceScan::ExclusiveSum(nullptr, bytesCheck, d_PBM, d_PBM_counts, BIN_COUNT + 1);
			if (bytesCheck > d_CUB_temp_storage_bytes)
			{
				if (d_CUB_temp_storage)
				{
					CUDA_CALL(cudaFree(d_CUB_temp_storage));
				}
				d_CUB_temp_storage_bytes = bytesCheck;
				CUDA_CALL(cudaMalloc(&d_CUB_temp_storage, d_CUB_temp_storage_bytes));
			}
		}

		float pbmMillis_control = 0, kernelMillis_control = 0;
		float pbmMillis = 0, kernelMillis = 0;
		for (unsigned int _j = 1; _j < UINT_MAX; --_j)
		{
			//1 = control
			//0 = smart FOV
			bool isControl = _j != 0;

			//For 200 iterations (to produce an average)
			const unsigned int ITERATIONS = 1;
			for (unsigned int i = 0; i < ITERATIONS; ++i)
			{
				//Reset each run of average model
#ifndef CIRCLES
				CUDA_CALL(cudaMemcpy(d_out, d_agents_init, sizeof(glm::vec4)*AGENT_COUNT, cudaMemcpyDeviceToDevice));
#endif	
				cudaEventRecord(start_PBM);
				{//Build atomic histogram
					CUDA_CALL(cudaMemset(d_PBM_counts, 0x00000000, (BIN_COUNT + 1) * sizeof(unsigned int)));
					int blockSize;   // The launch configurator returned block size 
					CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram, 32, 0));//Randomly 32
																												 // Round up according to array size
					int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
					atomicHistogram << <gridSize, blockSize >> > (d_keys, d_vals, d_PBM_counts, d_out);
					CUDA_CALL(cudaDeviceSynchronize());
				}
				{//Scan (sum), to finalise PBM
					cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_PBM_counts, d_PBM, BIN_COUNT + 1);
				}
				{//Reorder messages
					int blockSize;   // The launch configurator returned block size 
					CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
																														 // Round up according to array size
					int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
					//Copy messages from d_messages to d_messages_swap, in hash order
					reorderLocationMessages << <gridSize, blockSize >> > (d_keys, d_vals, d_PBM, d_out, d_agents);
					CUDA_CHECK();
				}
				{//Fill PBM and Message Texture Buffers																			  
					CUDA_CALL(cudaDeviceSynchronize());//Wait for return
					CUDA_CALL(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec4) * AGENT_COUNT));
					CUDA_CALL(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (BIN_COUNT + 1)));
					CUDA_CALL(cudaDeviceSynchronize());//Wait for return
					////Unorder agents
					//CUDA_CALL(cudaMemcpy(d_agents, d_agents_init, sizeof(glm::vec4)*AGENT_COUNT, cudaMemcpyDeviceToDevice));
				}
				cudaEventRecord(end_PBM);
				cudaEventRecord(start_kernel);
				if (isControl)
				{
					//Each message samples radial neighbours (static model)
					int blockSize;   // The launch configurator returned block size 
					CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
																														 // Round up according to array size
					int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
					//Copy messages from d_agents to d_out, in hash order
					printf("control\n");
					neighbourSearch_control << <gridSize, blockSize >> > (d_agents_init, d_out);
					CUDA_CHECK();
				}
				else
				{
					//Each message samples radial neighbours (static model)
					int blockSize;   // The launch configurator returned block size 
					CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
																														 // Round up according to array size
					int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
					//Copy messages from d_agents to d_out, in hash order
					printf("Smart FOV\n");
					neighbourSearch << <gridSize, blockSize >> > (d_agents_init, d_out);
					CUDA_CHECK();
				}
				CUDA_CALL(cudaDeviceSynchronize());
				cudaEventRecord(end_kernel);
				cudaEventSynchronize(end_kernel);

				float _pbmMillis = 0, _kernelMillis = 0;
				cudaEventElapsedTime(&_pbmMillis, start_PBM, end_PBM);
				cudaEventElapsedTime(&_kernelMillis, start_kernel, end_kernel);
				if (isControl)
				{
					pbmMillis_control += _pbmMillis;
					kernelMillis_control += _kernelMillis;
				}
				else
				{
					pbmMillis += _pbmMillis;
					kernelMillis += _kernelMillis;
				}

			}//for(ITERATIONS)
			pbmMillis_control /= ITERATIONS;
			kernelMillis_control /= ITERATIONS;
			pbmMillis /= ITERATIONS;
			kernelMillis /= ITERATIONS;

			//{//Unorder messages
			//	int blockSize;   // The launch configurator returned block size 
			//	CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
			//																										 // Round up according to array size
			//	int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
			//	//Copy messages from d_out to d_agents, in hash order
			//	unsortMessages << <gridSize, blockSize >> > (d_keys, d_vals, d_PBM, d_out, d_agents);
			//	CUDA_CHECK();
			//	//Swap d_out and d_agents
			//	{
			//		glm::vec4 *t = d_out;
			//		d_out = d_agents;
			//		d_agents = t;
			//	}
				//Wait for return
				CUDA_CALL(cudaDeviceSynchronize());
				//Copy back to relative host array (for validation)
				CUDA_CALL(cudaMemcpy(isControl ? h_out_control : h_out, d_out, sizeof(glm::vec4)*AGENT_COUNT, cudaMemcpyDeviceToHost));
				CUDA_CALL(cudaDeviceSynchronize());
			//}
		}//for(MODE)
		CUDA_CALL(cudaUnbindTexture(d_texPBM));
		CUDA_CALL(cudaUnbindTexture(d_texMessages));
		CUDA_CALL(cudaFree(d_PBM_counts));
		CUDA_CALL(cudaFree(d_PBM));
		//log();
		printf("Control:     PBM: %.2fms, Kernel: %.2fms\n", pbmMillis_control, kernelMillis_control);
		printf("Smart FOV: PBM: %.2fms, Kernel: %.2fms\n", pbmMillis, kernelMillis);
		unsigned int fails = 0;
#ifndef CIRCLES

		{//Validation
		 //Validate results for average model
		 //thrust::sort(thrust::cuda::par, d_out, d_out + AGENT_COUNT, vec2Compare());
		 //CUDA_CALL(cudaMemcpy(isControl ? h_out_control : h_out, d_out, sizeof(glm::vec2)*AGENT_COUNT, cudaMemcpyDeviceToHost));
			for (unsigned int i = 0; i < AGENT_COUNT; ++i)
			{
				assert(!(isnan(h_out[i].x) || isnan(h_out[i].y)));
				if (isnan(h_out[i].x) || isnan(h_out[i].y))
					printf("err nan\n");
				auto ret = glm::epsilonEqual(h_out[i], h_out_control[i], EPSILON);
				if (!(ret.x&&ret.y&&ret.z&&ret.w))
				{
					if (fails == 0)
					{
						printf("#%d(%.5f, %.5f, %.5f, %.5f) vs (%.5f, %.5f, %.5f, %.5f)\n", i, 
							h_out_control[i].x, h_out_control[i].y, h_out_control[i].z, h_out_control[i].w, 
							h_out[i].x, h_out[i].y, h_out[i].z, h_out[i].w);
					}
					fails++;
				}
			}
			if (fails > 0)
				printf("%d/%d (%.1f%%) Failed.\n", fails, AGENT_COUNT, 100 * (fails / (float)AGENT_COUNT));
		}
#endif
		log(f, AVERAGE_NEIGHBOURS, AGENT_COUNT, ENV_WIDTH, pbmMillis_control, kernelMillis_control, pbmMillis, kernelMillis, fails);
	}

	CUDA_CALL(cudaUnbindTexture(d_texMessages));
	CUDA_CALL(cudaFree(d_vals));
	CUDA_CALL(cudaFree(d_keys));
	CUDA_CALL(cudaFree(d_agents));
	CUDA_CALL(cudaFree(d_agents_init));
	CUDA_CALL(cudaFree(d_out));
	free(h_out);
	free(h_out_control);
}
void runAgents(std::ofstream &f, const unsigned int AGENT_COUNT, const float DENSITY)
{
	//density refers to approximate number of neighbours
	run(f, (unsigned int)sqrt(AGENT_COUNT / (DENSITY*2.86 / 9)), AGENT_COUNT);
}
int main()
{
	{
		std::ofstream f;
		createLog(f);
		assert(f.is_open());
		for (unsigned int i = 20000; i <= 3000000; i += 20000)
		//for (unsigned int i = 160000; i <= 160000; i++)
		{
			for (unsigned int j = 10; j <= 150; j+=10)
			//for(unsigned int j = 10;j <= 10; j++)
			{
				//Run i agents in a density with roughly 60 radial neighbours, and log
				//Within this, it is tested over a range of proportional bin widths
				runAgents(f, i, (float)j);
			}
			//break;
		}
	}
	printf("fin\n");
	cudaDeviceReset();
	getchar();
	return 0;
}

