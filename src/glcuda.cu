#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "sphere.cuh"
#include <GL/glut.h>
#include<gl/gl.h>
#include<gl/GLU.h>

#define PI 3.1415926536
#define GRIDSPHERES 27
const int SPACESIZE = 10; // 空间大小
Sphere* spheres;
Sphere* d_spheres;
const int SPHERE_NUMBER = 64*4; // 小球总数
const float TIMEPERFRAME = 0.1; // 场景快慢
const float gravity = -0.07 * TIMEPERFRAME; // 重力大小
#define collisionEpsilon 0.08
int* gridContainSphereIndex;
int* gridContainSphereNumber;
int* d_gridContainSphereIndex;
int* d_gridContainSphereNumber;

__host__ __device__ float myMax(float a, float b)
{
    return a > b ? a : b;
}
__host__ __device__ float myMin(float a, float b)
{
    return a > b ? b : a;
}
__host__ __device__ int getIndexGCSN(int a, int b, int c)
{
    return c + b * SPACESIZE + a * SPACESIZE * SPACESIZE;
}
__host__ __device__ int getIndexGCSI(int a, int b, int c, int d)
{
    return d + c * GRIDSPHERES + b * GRIDSPHERES * SPACESIZE + a * GRIDSPHERES * SPACESIZE * SPACESIZE;
}

// 空间划分
__global__ void sphereGridIndex(Sphere* d_spheres, int* d_gridContainSphereIndex, int* d_gridContainSphereNumber, int SPHERE_NUMBER)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < SPHERE_NUMBER; i += stride)
    {
        Vec3f location(d_spheres[i].center);
        int geshu = atomicAdd(&(d_gridContainSphereNumber[getIndexGCSN(abs((int)location.x()),abs((int)location.y()),abs((int)location.z()))]),1);
        d_gridContainSphereIndex[getIndexGCSI(abs((int)location.x()), abs((int)location.y()), abs((int)location.z()), geshu)] = i;
    }
}
// 碰撞检测
__global__ void sphereGridCollision(Sphere* d_spheres, int* d_gridContainSphereIndex, int* d_gridContainSphereNumber, int SPHERE_NUMBER, float gravity)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < SPHERE_NUMBER; i += stride)
    {
        d_spheres[i].speed += Vec3f(0, gravity, 0);
        Sphere& theSphere = (d_spheres[i]);
        Vec3f location(d_spheres[i].center);
        float radius = d_spheres[i].radius;
        Vec3f& speed = (d_spheres[i].speed);
        float restitution = d_spheres[i].restitution;
        float mass = theSphere.mass;
        // 与墙碰撞
        if (location[0] - radius <= 0)
        {
            d_spheres[i].speed.Set(abs(restitution * speed[0]), speed[1], speed[2]);
            speed = d_spheres[i].speed;
        }
        if (location[1] - radius <= 0)
        {
            d_spheres[i].speed.Set(speed[0], abs(restitution * speed[1]), speed[2]);
            speed = d_spheres[i].speed;
        }
        if (location[2] - radius <= 0)
        {
            d_spheres[i].speed.Set(speed[0], speed[1], abs(restitution * speed[2]));
            speed = d_spheres[i].speed;
        }
        if (location[0] + radius >= SPACESIZE)
        {
            d_spheres[i].speed.Set(-abs(restitution * speed[0]), speed[1], speed[2]);
            speed = d_spheres[i].speed;
        }
        if (location[1] + radius >= SPACESIZE)
        {
            d_spheres[i].speed.Set(speed[0], -abs(restitution * speed[1]), speed[2]);
            speed = d_spheres[i].speed;
        }
        if (location[2] + radius >= SPACESIZE)
        {
            d_spheres[i].speed.Set(speed[0], speed[1], -abs(restitution * speed[2]));
            speed = d_spheres[i].speed;
        }

        //与其他球碰撞
        for (int m = myMax(location.x()-1, 0); m <= myMin(location.x() + 1, SPACESIZE-1); m++)
        {
            for (int n = myMax(location.y() - 1, 0); n <= myMin(location.y() + 1, SPACESIZE - 1); n++)
            {
                for (int o = myMax(location.z() - 1, 0); o <= myMin(location.z() + 1, SPACESIZE - 1); o++)
                {
                    //printf("d_gridContainSphereNumber[getIndexGCSN(m, n, o)]: %d\n", d_gridContainSphereNumber[getIndexGCSN(m, n, o)]);
                    for (int p = 0; p < d_gridContainSphereNumber[getIndexGCSN(m, n, o)]; p++)
                    {
                        int aimIndex = d_gridContainSphereIndex[getIndexGCSI(m, n, o, p)];
                        //printf("aimIndex%d, i%d\n", aimIndex, i);
                        if (aimIndex <= i) { continue; }
                        Sphere aimSphere = d_spheres[aimIndex];

                        Vec3f centerDistance(aimSphere.center - d_spheres[i].center);
                        //printf("pos0.0");
                        if (centerDistance.Length() > theSphere.radius + aimSphere.radius + collisionEpsilon) { continue; }
                        //printf("begin collision!");
                        // 引发碰撞


                        Vec3f aimSpeed(aimSphere.speed);
                        float aimMass = aimSphere.mass;
                        float collisionRestitution = (restitution + aimSphere.restitution) / 2;
                        
                        // 球心方向分速度
                        float fenSpeed = speed.Dot3(centerDistance) / centerDistance.Length();
                        float aimFenSpeed = aimSpeed.Dot3(centerDistance) / centerDistance.Length();

                        float fenSpeedFinal = (mass * fenSpeed + aimMass * aimFenSpeed + aimMass * collisionRestitution * (aimFenSpeed - fenSpeed) ) / (mass + aimMass);
                        float aimFenSpeedFinal = (mass * fenSpeed + aimMass * aimFenSpeed + mass * collisionRestitution * (fenSpeed - aimFenSpeed)) / (mass + aimMass);

                        float fenSpeedChangeB = fenSpeedFinal - fenSpeed; 
                        float aimFenSpeedChangeB = aimFenSpeedFinal - aimFenSpeed;
                        //printf("%f fenSpeed:%f, aimFenSpeed:%f, fenSpeedFinal:%f\n", fenSpeedFinal - fenSpeed, fenSpeed, aimFenSpeed, fenSpeedFinal);
                        Vec3f fenChangeSpeed(centerDistance * (fenSpeedChangeB / centerDistance.Length()));
                        Vec3f aimFenChangeSpeed(centerDistance * (aimFenSpeedChangeB / centerDistance.Length()));
                        /*printf("%f the:%f, %f, %f. aim: %f, %f, %f. change: %f, %f, %f\n" , fenSpeedFinal - fenSpeed,theSphere.center.x(), theSphere.center.y(), theSphere.center.z(),
                            aimSphere.center.x(), aimSphere.center.y(), aimSphere.center.z(), fenChangeSpeed.x(), fenChangeSpeed.y(), fenChangeSpeed.z());*/
                        d_spheres[i].speed += fenChangeSpeed;
                        d_spheres[aimIndex].speed += aimFenChangeSpeed;
                        speed = d_spheres[i].speed;
                    }
                }
            }
        }
    }
}
// 小球移动
__global__ void sphereMove(Sphere* d_spheres,int SPHERE_NUMBER, float TIMEPERFRAME)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < SPHERE_NUMBER; i += stride)
    {
        d_spheres[i].center += d_spheres[i].speed * TIMEPERFRAME;
    }
}
void initScene()
{
    spheres = (Sphere*)malloc(SPHERE_NUMBER * sizeof(Sphere));
    gridContainSphereIndex = (int *)malloc(SPACESIZE * SPACESIZE * SPACESIZE * GRIDSPHERES * sizeof(int));
    gridContainSphereNumber = (int *)malloc(SPACESIZE * SPACESIZE * SPACESIZE * sizeof(int));
    cudaMalloc((void**)&d_spheres, SPHERE_NUMBER * sizeof(Sphere));
    cudaMalloc((void**)&d_gridContainSphereIndex, SPACESIZE*SPACESIZE*SPACESIZE * GRIDSPHERES * sizeof(int));
    cudaMalloc((void**)&d_gridContainSphereNumber, SPACESIZE * SPACESIZE * SPACESIZE * sizeof(int));
    for (int i = 0; i < SPHERE_NUMBER; i++)
    {
        spheres[i].center.Set(1.0 + i % 64 % 8, 4 + i/64, 1.0 + i % 64 / 8);
        spheres[i].radius = 0.25 + (rand() % 1000) / 1000.0 * 0.25;
        spheres[i].color.Set((rand() % 1000) / 1000.0, (rand() % 1000) / 1000.0, (rand() % 1000) / 1000.0);
        spheres[i].speed.Set((rand() % 1000) / 1000.0, (rand() % 1000) / 1000.0, (rand() % 1000) / 1000.0);
        spheres[i].restitution = 0.8 + (rand() % 1000) / 1000.0 * 0.2; 
        spheres[i].mass = spheres[i].radius * spheres[i].radius * spheres[i].radius;
    }
}


void drawSphere(GLfloat xx, GLfloat yy, GLfloat zz, GLfloat radius, GLfloat M, GLfloat N)
{
    // 该函数参考网上博客，详情见文档参考文献
    float step_z = PI / M;
    float step_xy = 2 * PI / N;
    float x[4], y[4], z[4];

    float angle_z = 0.0;
    float angle_xy = 0.0;
    int i = 0, j = 0;
    glBegin(GL_QUADS);
    for (i = 0; i < M; i++)
    {
        angle_z = i * step_z;

        for (j = 0; j < N; j++)
        {
            angle_xy = j * step_xy;

            x[0] = radius * sin(angle_z) * cos(angle_xy);
            y[0] = radius * sin(angle_z) * sin(angle_xy);
            z[0] = radius * cos(angle_z);

            x[1] = radius * sin(angle_z + step_z) * cos(angle_xy);
            y[1] = radius * sin(angle_z + step_z) * sin(angle_xy);
            z[1] = radius * cos(angle_z + step_z);

            x[2] = radius * sin(angle_z + step_z) * cos(angle_xy + step_xy);
            y[2] = radius * sin(angle_z + step_z) * sin(angle_xy + step_xy);
            z[2] = radius * cos(angle_z + step_z);

            x[3] = radius * sin(angle_z) * cos(angle_xy + step_xy);
            y[3] = radius * sin(angle_z) * sin(angle_xy + step_xy);
            z[3] = radius * cos(angle_z);

            for (int k = 0; k < 4; k++)
            {
                glVertex3f(xx + x[k], yy + y[k], zz + z[k]);
            }
        }
    }
    glEnd();
}

void myDisplay()
{
    // 每一帧的碰撞检测和绘制
    //清楚缓冲区
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //保存当前模型视图矩阵。
    glPushMatrix();

    // 三面墙壁
    glColor3f(114/256.0, 83/256.0, 52/256.0);
    glBegin(GL_QUADS);
        glVertex3f(0,0,0);
        glVertex3f(SPACESIZE, 0, 0);
        glVertex3f(SPACESIZE, 0, SPACESIZE);
        glVertex3f(0, 0, SPACESIZE);
    glEnd();
    glColor3f(69 / 256.0, 137 / 256.0, 148 / 256.0);
    glBegin(GL_QUADS);
        glVertex3f(0, 0, 0);
        glVertex3f(0, SPACESIZE, 0);
        glVertex3f(SPACESIZE, SPACESIZE, 0);
        glVertex3f(SPACESIZE, 0, 0);
    glEnd();
    glColor3f(117 / 256.0, 121 / 256.0, 74 / 256.0);
    glBegin(GL_QUADS);
        glVertex3f(SPACESIZE, 0, 0);
        glVertex3f(SPACESIZE, 0, SPACESIZE);
        glVertex3f(SPACESIZE, SPACESIZE, SPACESIZE);
        glVertex3f(SPACESIZE, SPACESIZE, 0);
    glEnd();
    

    //开始运行碰撞检测相关代码
    dim3 blockSize(SPHERE_NUMBER);
    dim3 gridSize(1);
    
    memset(gridContainSphereIndex, 0, SPACESIZE * SPACESIZE * SPACESIZE * GRIDSPHERES * sizeof(int));
    memset(gridContainSphereNumber, 0, SPACESIZE * SPACESIZE * SPACESIZE * sizeof(int));

    cudaMemcpy((void*)d_gridContainSphereIndex, (void*)gridContainSphereIndex, SPACESIZE * SPACESIZE * SPACESIZE * GRIDSPHERES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_gridContainSphereNumber, (void*)gridContainSphereNumber, SPACESIZE * SPACESIZE * SPACESIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_spheres, (void*)spheres, SPHERE_NUMBER * sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    sphereGridIndex << < gridSize, blockSize >> > (d_spheres, d_gridContainSphereIndex, d_gridContainSphereNumber, SPHERE_NUMBER);

    
    cudaDeviceSynchronize();
    sphereGridCollision << < gridSize, blockSize >> > (d_spheres, d_gridContainSphereIndex, d_gridContainSphereNumber, SPHERE_NUMBER, gravity);

    cudaDeviceSynchronize();
    sphereMove << < gridSize, blockSize >> > (d_spheres, SPHERE_NUMBER, TIMEPERFRAME);
    cudaDeviceSynchronize();
   
    cudaMemcpy((void*)spheres, (void*)d_spheres, SPHERE_NUMBER * sizeof(Sphere), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    //绘制所有小球
    for (int i = 0; i < SPHERE_NUMBER; i++)
    {
        glColor3f(spheres[i].color[0], spheres[i].color[1], spheres[i].color[2]);
        drawSphere(spheres[i].center[0], spheres[i].center[1], spheres[i].center[2], spheres[i].radius, 10, 10);
    }

    // 弹出堆栈
    glPopMatrix();

    // 交换缓冲区
    glutSwapBuffers();
}
void changeSize(int w, int h) {

    // 防止除数即高度为0
    if (h == 0)
        h = 1;

    float ratio = 1.0 * w / h;

    // 单位化投影矩阵。
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // 设置视口大小为增个窗口大小
    glViewport(0, 0, w, h);

    // 设置正确的投影矩阵
    gluPerspective(45, ratio, 1, 1000);
    //下面是设置模型视图矩阵
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(-8.0, 10.0, 18.0, 5.0, 3.5, 5.0, 0.0f, 1.0f, 0.0f);
}

int main(int argc, char* argv[])
{
    // 程序入口，完成opengl初始化，建立窗口
    srand(1);
    initScene();
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 10);
    glutInitWindowSize(800, 800);
    glutCreateWindow("spheerCollision");

    glutDisplayFunc(&myDisplay);

    glutIdleFunc(myDisplay);

    glutReshapeFunc(changeSize);

    glEnable(GL_DEPTH_TEST);
    glutMainLoop();

    return 0;

}