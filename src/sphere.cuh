// 定义了小球数据结构
#include "util_vectors.cuh"
class Sphere
{
public:
	Vec3f center;
	float radius;
	Vec3f color;
	Vec3f speed;
	float restitution;
	float mass;
};