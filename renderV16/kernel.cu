#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <vector>


struct Vector3 {
    float X, Y, Z;
};

__device__ Vector3 Hit(Vector3 center, int radius, Vector3 origin, Vector3 direction, float zfar)
{
    Vector3 oc = { center.X - origin.X, center.Y - origin.Y, center.Z - origin.Z };
    float a = direction.X * direction.X + direction.Y * direction.Y + direction.Z * direction.Z;
    float half_b = direction.X * oc.X + direction.Y * oc.Y + direction.Z * oc.Z;
    float c = oc.X * oc.X + oc.Y * oc.Y + oc.Z * oc.Z - radius * radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0)
    {
        return { 0, 0, zfar };
    }
    float sqrtd = sqrt(discriminant);

    float root = (half_b - sqrtd) / a;
    float z1 = origin.Z + root * direction.Z;
    float y1 = origin.Y + root * direction.Y;
    float x1 = origin.X + root * direction.X;

    root = (half_b + sqrtd) / a;
    float z2 = origin.Z + root * direction.Z;
    float y2 = origin.Y + root * direction.Y;
    float x2 = origin.X + root * direction.X;

    if (z1 >= z2) return { x1, y1, z1 };
    else return { x2, y2, z2 };
}

extern "C" __global__ void RenderSpheres(float* thing, float* depth, unsigned char* canvas, int things, Vector3 camera, int radius, int cx, int cy, float zfar, Vector3 light1, Vector3 light2)
{
    float x, y, z, r, g, b;
    float xminP, yminP, xmaxP, ymaxP;
    int index;
    float shadow;
    float distance;
    float norX, norY, norZ;
    float ligX, ligY, ligZ;

    int chunksize = cx / blockDim.x;
    int xmin = chunksize * threadIdx.x;
    int xmax = chunksize * (threadIdx.x + 1);

    chunksize = cy / blockDim.y;
    int ymin = chunksize * threadIdx.y;
    int ymax = chunksize * (threadIdx.y + 1);

    for (int i = 0; i < things; i++)
    {
        x = thing[6 * i];
        y = thing[6 * i + 1];
        z = thing[6 * i + 2];
        r = thing[6 * i + 3];
        g = thing[6 * i + 4];
        b = thing[6 * i + 5];

        xminP = camera.X + (x + radius - camera.X) * (0 - camera.Z) / (z - camera.Z);
        yminP = camera.Y + (y + radius - camera.Y) * (0 - camera.Z) / (z - camera.Z);
        xmaxP = camera.X + (x - radius - camera.X) * (0 - camera.Z) / (z - camera.Z);
        ymaxP = camera.Y + (y - radius - camera.Y) * (0 - camera.Z) / (z - camera.Z);
        if (xminP < xmin || xmaxP >= xmax || yminP < ymin || ymaxP >= ymax) continue;
        
        for (int yP = ymin; yP < ymax; yP++)
        {
            for (int xP = xmin; xP < xmax; xP++)
            {
                index = xP + yP * cx;

                Vector3 center = { x, y, z };
                Vector3 direction = { xP - camera.X, yP - camera.Y, 0 - camera.Z };
                Vector3 ray = Hit(center, radius, camera, direction, zfar);
                if (ray.Z > depth[3 * index])
                {
                    depth[3 * index] = ray.Z;

                    shadow = 0;
                    norX = ray.X - x;
                    norY = ray.Y - y;
                    norZ = ray.Z - z;
                    distance = sqrt(norX * norX + norY * norY + norZ * norZ);
                    norX /= distance;
                    norY /= distance;
                    norZ /= distance;
                    ligX = xP - light1.X;
                    ligY = yP - light1.Y;
                    ligZ = 0 - light1.Z;
                    distance = sqrt(ligX * ligX + ligY * ligY + ligZ * ligZ);
                    ligX /= distance;
                    ligY /= distance;
                    ligZ /= distance;
                    shadow += -24.0f * (1 + norX * ligX + norY * ligY + norZ * ligZ);
                    ligX = xP - light2.X;
                    ligY = yP - light2.Y;
                    ligZ = 0 - light2.Z;
                    distance = sqrt(ligX * ligX + ligY * ligY + ligZ * ligZ);
                    ligX /= distance;
                    ligY /= distance;
                    ligZ /= distance;
                    shadow += -24.0f * (1 + norX * ligX + norY * ligY + norZ * ligZ);

                    canvas[4 * index] = (unsigned char)(b + shadow);
                    canvas[4 * index + 1] = (unsigned char)(g + shadow);
                    canvas[4 * index + 2] = (unsigned char)(r + shadow);
                    canvas[4 * index + 3] = 255;
                }
            }
        }
    }
}

extern "C" __global__ void RenderLight(float* thing, float* depth, unsigned char* canvas, int things, int channel, Vector3 light, int radius, int cx, int cy, float zfar)
{
    float x, y, z;
    float xminP, yminP, xmaxP, ymaxP;
    int index;

    int chunksize = cx / blockDim.x;
    int xmin = chunksize * threadIdx.x;
    int xmax = chunksize * (threadIdx.x + 1);

    chunksize = cy / blockDim.y;
    int ymin = chunksize * threadIdx.y;
    int ymax = chunksize * (threadIdx.y + 1);

    for (int i = 0; i < things; i++)
    {
        x = thing[6 * i];
        y = thing[6 * i + 1];
        z = thing[6 * i + 2];

        xminP = light.X + (x + radius - light.X) * (0 - light.Z) / (z - light.Z);
        yminP = light.Y + (y + radius - light.Y) * (0 - light.Z) / (z - light.Z);
        xmaxP = light.X + (x - radius - light.X) * (0 - light.Z) / (z - light.Z);
        ymaxP = light.Y + (y - radius - light.Y) * (0 - light.Z) / (z - light.Z);
        if (xminP < xmin || xmaxP >= xmax || yminP < ymin || ymaxP >= ymax) continue;

        for (int yP = ymin; yP < ymax; yP++)
        {
            for (int xP = xmin; xP < xmax; xP++)
            {
                index = xP + yP * cx;

                Vector3 center = { x, y, z };
                Vector3 direction = { xP - light.X, yP - light.Y, 0 - light.Z };
                Vector3 ray = Hit(center, radius, light, direction, zfar);
                if (ray.Z > depth[3 * index + channel])
                {
                    depth[3 * index + channel] = ray.Z;
                }
            }
        }
    }
}
