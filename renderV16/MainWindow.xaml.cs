using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using System.Numerics;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media.Imaging;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using Windows.Graphics.Imaging;
using Windows.Storage.Streams;
using Windows.Storage;
using Windows.Storage.Pickers;
using WinRT.Interop;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing;
using System.Runtime.InteropServices;
using g3;

namespace renderV16
{
    public sealed partial class MainWindow : Window
    {
        // System variables
        DispatcherTimer timer = new DispatcherTimer();
        Random rnd = new Random();
        string[] log = new string[30];
        int state = 0;

        // Variables
        Vector3 camera;           // Camera position
        Vector3 light1, light2;   // Light position
        WriteableBitmap bitmap;   // Image bitmap to stream to screen
        List<Thing> thing;        // Spheres
        float[] thingarray;       // Spheres array (for Cuda)
        float[] depth;            // Z depth array for camera, light1, light2 (for Cuda)
        byte[] canvas;            // Bytes array for screen (for Cuda)
        byte[] wall;              // Bytes array for background
        int wx, wy;               // Screen or Wall dimensions
        int fx0, fy0, fx, fy;     // Canvas frame dimensions
        int cx0, cy0, cx, cy, cz; // Canvas dimensions including shadow area
        float zfocal;             // z distance from camera to canvas bottom
        float zfar;               // z distance from gallery wall (0) to lowest z (canvas bottom) 
        float znear;              // z distance from gallery wall to highest z
        bool solid;               // Parameter to draw frame walls
        float[] parameters;       // List of parameters to fit frames within different wall types
        IReadOnlyList<StorageFile> files; // List of images or frames loaded from disc
        int frames, frame;        // Total number of frames and current frame id 

        // Parameters
        readonly int radius = 4;    // Sphere radius
        readonly float coarse = 5;  // Space between samples taken from frames to create things or spheres
        readonly float zoom = 1f;   // Image render quality
        readonly string wallname = "wall06"; // Type of wall chosen

        //Cuda Variables and Kernels
        static CudaDeviceVariable<float> devThing;
        static CudaDeviceVariable<float> devDepth;
        static CudaDeviceVariable<byte> devCanvas;
        static CudaKernel RenderSpheresKernel;
        static CudaKernel RenderLightKernel;
        static CudaContext cudaContext;

        // Dictionary to store items and their float parameters
        // Parameters are { wallx, wally, canvas_x0, canvas_y0, canvasx, canvasy,
        // zfocal, zfar, camera_height, light1x, light1y, light1z, light2x, light2y, light2z, margin, solid_frame } 
        static Dictionary<string, float[]> items = new Dictionary<string, float[]>
        {
        { "wall01", new float[] { 1365, 768, 500, 30, 512, 650, 5, -0f, 0.66f, 0.2f, 0.2f, 1f, 0.33f, 0.33f, 1f, 1 } },
        { "wall02", new float[] { 1365, 768, 100, 0, 512, 615, 3, -0.75f, 0.55f, 0.7f, 0.1f, 1f, 0.99f, 0.5f, 0.6f, 1 } },
        { "wall03", new float[] { 768, 1365, 99, 192, 472, 722, 3, -0.5f, 0.5f, 0.7f, 0.1f, 1f, 0.99f, 0.5f, 0.6f, 1 } },
        { "wall04", new float[] { 1365, 768, 730, 110, 438, 438, 7, -0.75f, 0.4f, 0.4f, 0f, 0.5f, 0.99f, 0.5f, 0.3f, 1 } },
        { "wall05", new float[] { 1365, 768, 0, 65, 1365, 590, 3.7f, -1f, 0.6f, 0.4f, 0.1f, 0.4f, 0.6f, 0.1f, 0.1f, 0 } },
        { "wall06", new float[] { 1365, 768, 418, 79, 528, 551, 6f, -0.75f, 0.55f, 0.1f, 0.2f, 0.5f, 0f, 0.5f, 0.6f, 1 } },
        { "wall07", new float[] { 1365, 768, 316, 18, 732, 732, 6, -0.75f, 0.5f, 0f, 0f, 0.5f, 0f, 0.5f, 0.6f, 1 } },
        { "wall08", new float[] { 1365, 768, 0, 0, 1365, 620, 11, -1f, 0.4f, 0.9f, 0f, 1f, 0.99f, 0.5f, 0.6f , 1 } },
        { "wall10", new float[] { 1365, 768, 0, 20, 1315, 600, 3.7f, -1f, 0.63f, 0.9f, 0f, 1f, 0.99f, 0.5f, 0.6f, 1 } },
        { "wall11", new float[] { 1365, 768, 0, 0, 1335, 630, 12f, -1f, 0.4f, 0.9f, 0f, 1f, 0.99f, 0.5f, 0.6f, 1 } },
        { "wall12", new float[] { 1365, 768, 366, 65, 635, 533, 3f, -0.5f, 0.5f, 0.99f, 0f, 1f, 0.99f, 0.1f, 0.6f, 1 } },
        { "wall13", new float[] { 768, 1365, 243, 710, 299, 517, 3f, -0.5f, 0.9f, 0f, 0f, 1f, 0.99f, 0.5f, 0.6f, 1 } },
        { "wall14", new float[] { 1365, 768, 393, 140, 546, 399, 5f, -0.75f, 0.5f, 0.9f, 0.1f, 1f, 0.9f, 0.1f, 1f, 1 } },
        { "wall20", new float[] { 1365, 768, 518, 95, 355, 369, 7f, -0.75f, 0.5f, 0.9f, 0.1f, 1f, 0.4f, 0.1f, 1f, 1 } },
        { "wall21", new float[] { 1365, 768, 459, 49, 447, 555, 8f, -0.75f, 0.6f, 0.7f, 0f, 0.8f, 0f, 0.5f, 0.6f, 1 } }
        };


        public MainWindow()
        {
            InitializeComponent();

            string selectedItem = wallname;
            parameters = LoadParameters(selectedItem);

            // Setup with chosen parameters
            if (parameters == null)
            {
                WriteLog("> No parameters found for the selected item.");
            }
            else
            {
                // Sets wall and canvas dimensions as a 'zoom' multiple screen coordinates for higher precision
                wx = (int)(zoom * parameters[0]);
                wy = (int)(zoom * parameters[1]);
                cx0 = 0;
                cy0 = 0;
                cx = (int)(zoom * parameters[0]);
                cy = (int)(zoom * parameters[1]);
                cz = (int)(zoom * 256);
                fx0 = (int)(zoom * parameters[2]);
                fy0 = (int)(zoom * parameters[3]);
                fx = (int)(zoom * parameters[4]);
                fy = (int)(zoom * parameters[5]);
                zfocal = cz * parameters[6];
                zfar = cz * parameters[7];
                znear = cz + zfar;
                camera = new Vector3(0.5f * cx, parameters[8] * cy, zfocal);
                light1 = new Vector3(parameters[9] * cx, parameters[10] * cy, parameters[11] * zfocal);
                light2 = new Vector3(parameters[12] * cx, parameters[13] * cy, parameters[14] * zfocal);
                if (parameters[15] > 0) solid = true;
                else solid = false;
                wall = new byte[4 * cx * cy];
                canvas = new byte[4 * cx * cy];
                depth = new float[3 * cx * cy];
            }

            timer.Tick += Timer_Tick;
            timer.Interval = new TimeSpan(0, 0, 0, 0, 1);
            timer.Start();
        }

        public async void Timer_Tick(object sender, object e)
        {
            // Loads background wall
            if (state == 0)
            {
                state = -1;

                // Load wall texture to canvas
                string fname = string.Concat("C:\\Users\\nunoa\\OneDrive\\Pictures\\", wallname, ".png");
                wall = await LoadImage(fname, cx, cy);

                var openPicker = new FileOpenPicker();
                // Initialize the file picker with the window handle (HWND)
                var hWnd = WindowNative.GetWindowHandle(this);
                InitializeWithWindow.Initialize(openPicker, hWnd);
                // Set options for the file picker
                openPicker.ViewMode = PickerViewMode.Thumbnail;
                openPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
                openPicker.FileTypeFilter.Add(".png");
                // Open the picker for the user to pick multiple files
                files = await openPicker.PickMultipleFilesAsync();
                frames = files.Count;
                frame = 0;
                if (files != null && files.Count > 0)
                {
                    WriteLog($"> Read {frames} image");
                }

                // Cuda setup at maximmum of needed things
                int max = (int)(512 * 512 / (Math.Sqrt(3) / 2)) + 1;
                thingarray = new float[6 * max];
                cudaContext = new CudaContext();
                devThing = new CudaDeviceVariable<float>(sizeof(float) * 6 * max);
                devDepth = new CudaDeviceVariable<float>(sizeof(float) * 3 * cx * cy);
                devCanvas = new CudaDeviceVariable<byte>(sizeof(byte) * 4 * cx * cy);

                state = 1;
            }

            // Loads frames and creates things with image pigment RGB taken at XY position
            // Z depth is calculated from color luminance for 3D transformation
            if (state == 1)
            {
                state = -1;

                // Reads input image
                (var image, var imagex, var imagey) = await ReadImage(files[frame]);
                // One can take a detail of the image instead of the full image i.e. zoom frame by 2x starting at 10% x 10% position
                image = ResizeWithBicubicInterpolation(image, imagex, imagey, 2f, 0.1f, 0.1f);
                // Make sure frame fits the canvas of the loaded wall
                (var pixels, var px, var py) = CropImageToFrame(image, imagex, imagey, (int)(fx / zoom), (int)(fy / zoom));
                pixels = ApplyVignetteToImage(pixels, px, py);

                // Creates list of things by taking honeycomb samples from input image
                thing = new List<Thing>();
                GetThings(pixels, px, py, fx0, fy0, fx, fy, cz, zfar, (int)(zoom * radius), zoom, coarse);               

                state = 2;              
            }

            // Render and save frame
            if (state == 2)
            {
                state = -1;

                Reset(depth, camera, light1, light2, cx, cy, fx0, fy0, fx, fy, solid);
                RenderSpheres(camera);
                RenderLight(1, light1);
                RenderLight(2, light2);
                RenderShadows();

                bitmap = new WriteableBitmap(cx, cy);
                using (Stream stream = bitmap.PixelBuffer.AsStream())
                {
                    await stream.WriteAsync(canvas, 0, canvas.Length);
                }
                image.Source = bitmap;
                SoftwareBitmap outputBitmap = SoftwareBitmap.CreateCopyFromBuffer(
                    bitmap.PixelBuffer,
                    BitmapPixelFormat.Bgra8,
                    bitmap.PixelWidth,
                    bitmap.PixelHeight);
                await SaveImage(outputBitmap, $"Frame{frame:D4}_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                WriteLog($"> Saved frame {frame}");

                frame++;
                if (frame >= frames)
                {
                    WriteLog("> Done");
                    state = -1;
                }
                else state = 1;
            }
        }

        static float[] LoadParameters(string selectedItem)
        {
            if (items.TryGetValue(selectedItem, out float[] parameters))
            {
                return parameters;
            }
            return null;
        }

        public void Reset(float[] depth, Vector3 camera, Vector3 light1, Vector3 light2, int cx, int cy, int fx0, int fy0, int fx, int fy, bool solid)
        {
            int index;
            int pos;

            int x1 = (int)(camera.X + (fx0 - camera.X) * (0 - camera.Z) / (zfar - camera.Z));
            int y1 = (int)(camera.Y + (fy0 - camera.Y) * (0 - camera.Z) / (zfar - camera.Z));
            int x2 = (int)(camera.X + (fx0 + fx - camera.X) * (0 - camera.Z) / (zfar - camera.Z));
            int y2 = (int)(camera.Y + (fy0 + fy - camera.Y) * (0 - camera.Z) / (zfar - camera.Z));
            for (int yP = 0; yP < cy; yP++)
            {
                for (int xP = 0; xP < cx; xP++)
                {
                    index = xP + yP * cx;
                    pos = (cx0 + xP) + (cy0 + yP) * wx;
                    if (xP < fx0 || xP >= fx0 + fx || yP < fy0 || yP >= fy0 + fy)
                    {
                        depth[3 * index] = 0;
                        depth[3 * index + 1] = 0;
                        depth[3 * index + 2] = 0;
                        canvas[4 * index] = wall[4 * pos];
                        canvas[4 * index + 1] = wall[4 * pos + 1];
                        canvas[4 * index + 2] = wall[4 * pos + 2];
                        canvas[4 * index + 3] = 255;

                    }
                    else if (xP < x1 || xP > x2 || yP < y1 || yP > y2)
                    {
                        depth[3 * index] = zfar;
                        depth[3 * index + 1] = zfar;
                        depth[3 * index + 2] = zfar;
                        canvas[4 * index] = wall[4 * pos];
                        canvas[4 * index + 1] = wall[4 * pos + 1];
                        canvas[4 * index + 2] = wall[4 * pos + 2];
                        canvas[4 * index + 3] = 255;
                    }
                    else
                    {
                        depth[3 * index] = zfar;
                        depth[3 * index + 1] = zfar;
                        depth[3 * index + 2] = zfar;
                        canvas[4 * index] = (byte)(wall[4 * pos] * 0.2f);
                        canvas[4 * index + 1] = (byte)(wall[4 * pos + 1] * 0.2f);
                        canvas[4 * index + 2] = (byte)(wall[4 * pos + 2] * 0.2f);
                        canvas[4 * index + 3] = 255;
                    }
                }
            }

            for (int z = 0; z >= zfar; z--)
            {
                for (int y = fy0; y <= fy0 + fy; y++)
                {
                    int x = fx0;
                    int xP = (int)(camera.X + (x - camera.X) * (0 - camera.Z) / (z - camera.Z));
                    int yP = (int)(camera.Y + (y - camera.Y) * (0 - camera.Z) / (z - camera.Z));
                    index = xP + yP * wx;

                    if (z > depth[3 * index + 0])
                    {
                        depth[3 * index + 0] = z;

                        if (solid)
                        {
                            float shadow = 0;
                            Vector3 normal = new Vector3(1, 0, 0);
                            Vector3 light = Vector3.Normalize(new Vector3(-(x - light1.X), -(y - light1.Y), -(z - light1.Z)));
                            shadow += 32.0f * (Vector3.Dot(normal, light) - 1);
                            light = Vector3.Normalize(new Vector3(-(x - light2.X), -(y - light2.Y), -(z - light2.Z)));
                            shadow += 32.0f * (Vector3.Dot(normal, light) - 1);
                            canvas[4 * index + 2] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                            canvas[4 * index + 1] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                            canvas[4 * index + 0] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                        }
                    }
                    xP = (int)(light1.X + (x - light1.X) * (0 - light1.Z) / (z - light1.Z));
                    yP = (int)(light1.Y + (y - light1.Y) * (0 - light1.Z) / (z - light1.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 1])
                    {
                        depth[3 * index + 1] = z;
                    }
                    xP = (int)(light2.X + (x - light2.X) * (0 - light2.Z) / (z - light2.Z));
                    yP = (int)(light2.Y + (y - light2.Y) * (0 - light2.Z) / (z - light2.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 2])
                    {
                        depth[3 * index + 2] = z;
                    }

                    x = fx0 + fx;
                    xP = (int)(camera.X + (x - camera.X) * (0 - camera.Z) / (z - camera.Z));
                    yP = (int)(camera.Y + (y - camera.Y) * (0 - camera.Z) / (z - camera.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 0])
                    {
                        depth[3 * index + 0] = z;

                        if (solid)
                        {
                            float shadow = 0;
                            Vector3 normal = new Vector3(-1, 0, 0);
                            Vector3 light = Vector3.Normalize(new Vector3(-(x - light1.X), -(y - light1.Y), -(z - light1.Z)));
                            shadow += 32.0f * (Vector3.Dot(normal, light) - 1);
                            light = Vector3.Normalize(new Vector3(-(x - light2.X), -(y - light2.Y), -(z - light2.Z)));
                            shadow += 32.0f * (Vector3.Dot(normal, light) - 1);
                            canvas[4 * index + 2] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                            canvas[4 * index + 1] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                            canvas[4 * index + 0] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                        }
                    }
                    xP = (int)(light1.X + (x - light1.X) * (0 - light1.Z) / (z - light1.Z));
                    yP = (int)(light1.Y + (y - light1.Y) * (0 - light1.Z) / (z - light1.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 1])
                    {
                        depth[3 * index + 1] = z;
                    }
                    xP = (int)(light2.X + (x - light2.X) * (0 - light2.Z) / (z - light2.Z));
                    yP = (int)(light2.Y + (y - light2.Y) * (0 - light2.Z) / (z - light2.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 2])
                    {
                        depth[3 * index + 2] = z;
                    }

                }
                for (int x = fx0; x <= fx0 + fx; x++)
                {
                    int y = fy0;
                    int xP = (int)(camera.X + (x - camera.X) * (0 - camera.Z) / (z - camera.Z));
                    int yP = (int)(camera.Y + (y - camera.Y) * (0 - camera.Z) / (z - camera.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 0])
                    {
                        depth[3 * index + 0] = z;

                        if (solid)
                        {
                            float shadow = 0;
                            Vector3 normal = new Vector3(0, 1, 0);
                            Vector3 light = Vector3.Normalize(new Vector3(-(x - light1.X), -(y - light1.Y), -(z - light1.Z)));
                            shadow += 32.0f * (Vector3.Dot(normal, light) - 1);
                            light = Vector3.Normalize(new Vector3(-(x - light2.X), -(y - light2.Y), -(z - light2.Z)));
                            shadow += 32.0f * (Vector3.Dot(normal, light) - 1);
                            canvas[4 * index + 2] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                            canvas[4 * index + 1] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                            canvas[4 * index + 0] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                        }
                    }
                    xP = (int)(light1.X + (x - light1.X) * (0 - light1.Z) / (z - light1.Z));
                    yP = (int)(light1.Y + (y - light1.Y) * (0 - light1.Z) / (z - light1.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 1])
                    {
                        depth[3 * index + 1] = z;
                    }
                    xP = (int)(light2.X + (x - light2.X) * (0 - light2.Z) / (z - light2.Z));
                    yP = (int)(light2.Y + (y - light2.Y) * (0 - light2.Z) / (z - light2.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 2])
                    {
                        depth[3 * index + 2] = z;
                    }

                    y = fy0 + fy;
                    xP = (int)(camera.X + (x - camera.X) * (0 - camera.Z) / (z - camera.Z));
                    yP = (int)(camera.Y + (y - camera.Y) * (0 - camera.Z) / (z - camera.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 0])
                    {
                        depth[3 * index + 0] = z;

                        if (solid)
                        {
                            float shadow = 0;
                            Vector3 normal = new Vector3(0, -1, 0);
                            Vector3 light = Vector3.Normalize(new Vector3(-(x - light1.X), -(y - light1.Y), -(z - light1.Z)));
                            shadow += 32.0f * (Vector3.Dot(normal, light) - 1);
                            light = Vector3.Normalize(new Vector3(-(x - light2.X), -(y - light2.Y), -(z - light2.Z)));
                            shadow += 32.0f * (Vector3.Dot(normal, light) - 1);
                            canvas[4 * index + 2] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                            canvas[4 * index + 1] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                            canvas[4 * index + 0] = (byte)Math.Max(0, 64 + rnd.Next(0, 40) + shadow);
                        }
                    }
                    xP = (int)(light1.X + (x - light1.X) * (0 - light1.Z) / (z - light1.Z));
                    yP = (int)(light1.Y + (y - light1.Y) * (0 - light1.Z) / (z - light1.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 1])
                    {
                        depth[3 * index + 1] = z;
                    }
                    xP = (int)(light2.X + (x - light2.X) * (0 - light2.Z) / (z - light2.Z));
                    yP = (int)(light2.Y + (y - light2.Y) * (0 - light2.Z) / (z - light2.Z));
                    index = xP + yP * wx;
                    if (z > depth[3 * index + 2])
                    {
                        depth[3 * index + 2] = z;
                    }
                }
            }
        }

        public void RenderSpheres(Vector3 source)
        {
            if (RenderSpheresKernel == null)
            {
                RenderSpheresKernel = cudaContext.LoadKernel("C:\\Users\\nunoa\\source\\repos\\renderV14\\renderV14\\kernel.ptx", "RenderSpheres");

                RenderSpheresKernel.BlockDimensions = new dim3(32, 32);
                RenderSpheresKernel.GridDimensions = new dim3(1);
            }

            for (int i = 0; i < thing.Count; i++)
            {
                thingarray[6 * i] = thing[i].X;
                thingarray[6 * i + 1] = thing[i].Y;
                thingarray[6 * i + 2] = thing[i].Z;
                thingarray[6 * i + 3] = thing[i].R;
                thingarray[6 * i + 4] = thing[i].G;
                thingarray[6 * i + 5] = thing[i].B;
            }
            devThing.CopyToDevice(thingarray);
            devDepth.CopyToDevice(depth);
            devCanvas.CopyToDevice(canvas);

            RenderSpheresKernel.Run(devThing.DevicePointer, devDepth.DevicePointer, devCanvas.DevicePointer, thing.Count,
                source, radius, cx, cy, zfar, light1, light2);

            devDepth.CopyToHost(depth);
            devCanvas.CopyToHost(canvas);
        }

        public void RenderLight(int channel, Vector3 source)
        {
            if (RenderLightKernel == null)
            {
                RenderLightKernel = cudaContext.LoadKernel("C:\\Users\\nunoa\\source\\repos\\renderV14\\renderV14\\kernel.ptx", "RenderLight");

                RenderLightKernel.BlockDimensions = new dim3(32, 32);
                RenderLightKernel.GridDimensions = new dim3(1);
            }

            devDepth.CopyToDevice(depth);

            RenderLightKernel.Run(devThing.DevicePointer, devDepth.DevicePointer, devCanvas.DevicePointer, thing.Count,
                channel, source, radius, cx, cy, zfar);

            devDepth.CopyToHost(depth);
        }

        public void RenderShadows()
        {
            float shadow;
            float x, y, z;
            int xL, yL;
            int index, indexL;

            for (int yP = 0; yP < cy; yP++)
            {
                for (int xP = 0; xP < cx; xP++)
                {
                    shadow = 0;
                    index = xP + yP * cx;
                    z = depth[3 * index];
                    y = camera.Y + (yP - camera.Y) * (z - camera.Z) / (0 - camera.Z);
                    x = camera.X + (xP - camera.X) * (z - camera.Z) / (0 - camera.Z);

                    xL = (int)(light1.X + (x - light1.X) * (0 - light1.Z) / (z - light1.Z));
                    yL = (int)(light1.Y + (y - light1.Y) * (0 - light1.Z) / (z - light1.Z));
                    if (xL >= 0 && xL < cx && yL >= 0 && yL < cy)
                    {
                        indexL = xL + yL * cx;
                        if (depth[3 * indexL + 1] > depth[3 * index]) shadow -= 64 * (depth[3 * indexL + 1] - depth[3 * index]) / cz;
                    }

                    xL = (int)(light2.X + (x - light2.X) * (0 - light2.Z) / (z - light2.Z));
                    yL = (int)(light2.Y + (y - light2.Y) * (0 - light2.Z) / (z - light2.Z));
                    if (xL >= 0 && xL < cx && yL >= 0 && yL < cy)
                    {
                        indexL = xL + yL * cx;
                        if (depth[3 * indexL + 2] > depth[3 * index]) shadow -= 64 * (depth[3 * indexL + 2] - depth[3 * index]) / cz;
                    }

                    canvas[4 * index] = (byte)Math.Max(0, canvas[4 * index] + shadow);
                    canvas[4 * index + 1] = (byte)Math.Max(0, canvas[4 * index + 1] + shadow);
                    canvas[4 * index + 2] = (byte)Math.Max(0, canvas[4 * index + 2] + shadow);
                    canvas[4 * index + 3] = 255;
                }
            }
        }

        public void GetThings(byte[] pixels, int px, int py, int fx0, int fy0, int fx, int fy, int cz, float zfar, float radius, float zoom, float coarse)
        {
            List<Vector3d> vertices = new List<Vector3d>();
            List<Index3i> triangles = new List<Index3i>();
            float triX = coarse * 1f;
            float triY = coarse * (float)Math.Sqrt(3) / 2;

            int sx1 = 0, sx2 = 0, sy = 0;
            for (float y = 0; y < py; y += triY)
            {
                if (sy % 2 == 0)
                {
                    sx1 = 0;
                    for (float x = 0; x < px; x += triX)
                    {
                        float xPos = fx0 + radius + x * (fx - 2 * radius) / px;
                        float yPos = fy0 + radius + y * (fy - 2 * radius) / py;
                        int index = 4 * ((int)x + (int)y * px);
                        float r = pixels[index + 2];
                        float g = pixels[index + 1];
                        float b = pixels[index];
                        float zPos = zfar + (cz - radius) * (0.299f * r + 0.587f * g + 0.114f * b) / 255f;

                        vertices.Add(new Vector3d(xPos, yPos, zPos));

                        sx1++;
                    }
                }
                else
                {
                    sx2 = 0;
                    for (float x = triX / 2; x < px; x += triX)
                    {
                        float xPos = fx0 + radius + x * (fx - 2 * radius) / px;
                        float yPos = fy0 + radius + y * (fy - 2 * radius) / py;
                        int index = 4 * ((int)x + (int)y * px);
                        float r = pixels[index + 2];
                        float g = pixels[index + 1];
                        float b = pixels[index];
                        float zPos = zfar + (cz - radius) * (0.299f * r + 0.587f * g + 0.114f * b) / 255f;

                        vertices.Add(new Vector3d(xPos, yPos, zPos));

                        sx2++;
                    }
                }
                sy++;
            }

            for (int j = 0; j < sy - 1; j++)
            {
                if (j % 2 == 0)
                {
                    for (int i = 0; i < sx1 - 1; i++)
                    {
                        int topLeft = i + (j / 2) * (sx1 + sx2);
                        int topRight = topLeft + 1;
                        int bottomLeft = i + (j / 2) * (sx1 + sx2) + sx1;
                        int bottomRight = bottomLeft + 1;

                        triangles.Add(new Index3i(topLeft, bottomLeft, topRight));
                        if (i < sx2 - 1) triangles.Add(new Index3i(topRight, bottomLeft, bottomRight));
                    }
                }
                else
                {
                    for (int i = 0; i < sx2 - 1; i++)
                    {
                        int topLeft = i + (j / 2) * (sx1 + sx2) + sx1;
                        int topRight = topLeft + 1;
                        int bottomLeft = i + (j / 2) * (sx1 + sx2) + sx1 + sx2;
                        int bottomRight = bottomLeft + 1;

                        triangles.Add(new Index3i(topLeft, bottomLeft, bottomRight));
                        if (i < sx1 - 1) triangles.Add(new Index3i(topLeft, bottomRight, topRight));
                    }
                }
            }

            DMesh3 mesh = new DMesh3();
            foreach (Vector3d v in vertices)
            {
                mesh.AppendVertex(v);
            }

            foreach (Index3i tri in triangles)
            {
                mesh.AppendTriangle(tri);
            }
            WriteLog($"> mesh vertex {mesh.VertexCount}");

            // Create a remesher instance
            Remesher remesher = new Remesher(mesh);
            // Set remeshing parameters
            var targetEdgeLength = 6;
            // Create a projection target to maintain the original surface shape
            //remesher.ProjectionMode = Remesher.TargetProjectionMode.AfterRefinement;
            //remesher.EnableParallelProjection = true;
            //remesher.SetExternalConstraints(new MeshConstraints());
            //remesher.EnableSmoothing = true;
            remesher.SetTargetEdgeLength(targetEdgeLength);
            remesher.SmoothType = Remesher.SmoothTypes.Uniform;
            remesher.EnableFlips = true;
            remesher.EnableCollapses = true;
            remesher.EnableSplits = true;
            remesher.MinEdgeLength = targetEdgeLength * 0.7;
            remesher.MaxEdgeLength = targetEdgeLength * 1.3;

            remesher.SmoothSpeedT = 0.5f;
            //remesher.PreventNormalFlips = true;
            //remesher.SetProjectionTarget(MeshProjectionTarget.Auto(mesh));
            MeshConstraintUtil.PreserveBoundaryLoops(remesher);

            // Perform remeshing
            for (int i = 0; i < 10; i++)
            {
                remesher.BasicRemeshPass();
            }

            /*
            float ratiox = (float)(fx - 2 * radius) / px;
            float ratioy = (float)(fy - 2 * radius) / py;
            for (int i = 0; i < mesh.VertexCount; i++)
            {
                // Process the vertex coordinates and scale to canvas size
                var x = (float)mesh.GetVertex(i).x;
                var y = (float)mesh.GetVertex(i).y;
                var z = (float)mesh.GetVertex(i).z;

                var pos = (int)((x - fx0) / ratiox) + (int)((y - fy0) / ratioy) * px;
                if ((int)((x - fx0) / ratiox) < px && (int)((y - fy0) / ratioy) < py)
                {
                    var r = pixels[4 * pos + 2];
                    var g = pixels[4 * pos + 1];
                    var b = pixels[4 * pos];
                    thing.Add(new Thing(thing.Count, x, y, z, r, g, b));
                }
            }
            WriteLog($"> things {thing.Count}");
            */

            // Create a new DMesh3 with a dense index space
            DMesh3 remeshedMesh = new DMesh3(remesher.Mesh, true);
            // Compact the mesh to remove any unused vertices or faces
            remeshedMesh.CompactInPlace();
            // Creates things
            WriteLog($"> remesh vertex {remeshedMesh.VertexCount}");

            float ratiox = (float)(fx - 2 * radius) / px;
            float ratioy = (float)(fy - 2 * radius) / py;
            for (int i = 0; i < remeshedMesh.VertexCount; i++)
            {
                // Process the vertex coordinates and scale to canvas size
                var x = (float)remeshedMesh.GetVertex(i).x;
                var y = (float)remeshedMesh.GetVertex(i).y;
                var z = (float)remeshedMesh.GetVertex(i).z;

                var pos = (int)((x - fx0) / ratiox) + (int)((y - fy0) / ratioy) * px;
                if ((int)((x - fx0) / ratiox) < px && (int)((y - fy0) / ratioy) < py)
                {
                    var r = pixels[4 * pos + 2];
                    var g = pixels[4 * pos + 1];
                    var b = pixels[4 * pos];
                    thing.Add(new Thing(thing.Count, x, y, z, r, g, b));
                }
            }
            WriteLog($"> things {thing.Count}");

        }
       

        // ***************
        // Bitmap handling
        // ***************      
        private static (int cropX, int cropY, float ratio) CalculateCropCoordinates(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
        {
            float sourceAspect = (float)sourceWidth / sourceHeight;
            float targetAspect = (float)targetWidth / targetHeight;
            int cropX, cropY, newWidth, newHeight;
            float ratio;

            if (sourceAspect > targetAspect)
            {
                // Crop width
                ratio = (float)targetHeight / sourceHeight;
                newHeight = sourceHeight;
                newWidth = (int)(newHeight * targetAspect);
                cropX = (sourceWidth - newWidth) / 2;
                cropY = 0;
            }
            else
            {
                // Crop height
                ratio = (float)targetWidth / sourceWidth;
                newWidth = sourceWidth;
                newHeight = (int)(newWidth / targetAspect);
                cropX = 0;
                cropY = (sourceHeight - newHeight) / 2;
            }
            return (cropX, cropY, ratio);
        }

        private static (byte[], int, int) CropImageToFrame(byte[] source, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
        {
            float sourceAspect = (float)sourceWidth / sourceHeight;
            float targetAspect = (float)targetWidth / targetHeight;
            int cropX, cropY, newWidth, newHeight;

            if (sourceAspect > targetAspect)
            {
                // Crop width
                newHeight = sourceHeight;
                newWidth = (int)(newHeight * targetAspect);
                cropX = (sourceWidth - newWidth) / 2;
                cropY = 0;
            }
            else
            {
                // Crop height
                newWidth = sourceWidth;
                newHeight = (int)(newWidth / targetAspect);
                cropX = 0;
                cropY = (sourceHeight - newHeight) / 2;
            }

            byte[] crop = new byte[4 * newWidth * newHeight];

            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    int index1 = 4 * (x + y * newWidth);
                    int index2 = 4 * (x + cropX + (y + cropY) * sourceWidth);
                    crop[index1] = source[index2];
                    crop[index1 + 1] = source[index2 + 1];
                    crop[index1 + 2] = source[index2 + 2];
                    crop[index1 + 3] = 255;
                }
            }
            return (crop, newWidth, newHeight);
        }

        public static byte[] ResizeWithBicubicInterpolation(byte[] imageData, int originalWidth, int originalHeight, float scaleFactor, float scaleX0, float scaleY0)
        {
            int newWidth = (int)(originalWidth * scaleFactor);
            int newHeight = (int)(originalHeight * scaleFactor);

            using (Bitmap originalBitmap = new Bitmap(originalWidth, originalHeight, PixelFormat.Format32bppArgb))
            {
                BitmapData bmpData = originalBitmap.LockBits(new Rectangle(0, 0, originalWidth, originalHeight), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
                Marshal.Copy(imageData, 0, bmpData.Scan0, imageData.Length);
                originalBitmap.UnlockBits(bmpData);

                using (Bitmap resizedImage = new Bitmap(newWidth, newHeight, PixelFormat.Format32bppArgb))
                {
                    resizedImage.SetResolution(originalBitmap.HorizontalResolution, originalBitmap.VerticalResolution);

                    using (Graphics g = Graphics.FromImage(resizedImage))
                    {
                        g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                        g.CompositingMode = CompositingMode.SourceCopy;
                        g.CompositingQuality = CompositingQuality.HighQuality;
                        g.SmoothingMode = SmoothingMode.HighQuality;
                        g.PixelOffsetMode = PixelOffsetMode.HighQuality;

                        using (ImageAttributes wrapMode = new ImageAttributes())
                        {
                            wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                            g.DrawImage(originalBitmap, new Rectangle(0, 0, newWidth, newHeight), 0, 0, originalWidth, originalHeight, GraphicsUnit.Pixel, wrapMode);
                        }
                    }

                    BitmapData resizedData = resizedImage.LockBits(new Rectangle(0, 0, newWidth, newHeight), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
                    byte[] resizedBytes = new byte[resizedData.Stride * resizedData.Height];
                    Marshal.Copy(resizedData.Scan0, resizedBytes, 0, resizedBytes.Length);
                    resizedImage.UnlockBits(resizedData);

                    byte[] cropped = new byte[4 * originalWidth * originalHeight];
                    int index, scaledIndex;

                    for (int y = 0; y < originalHeight; y++)
                    {
                        for (int x = 0; x < originalWidth; x++)
                        {
                            index = 4 * x + 4 * y * originalWidth;
                            scaledIndex = 4 * (int)(scaleX0 * newWidth + x) + 4 * (int)(scaleY0 * newHeight + y) * newWidth;
                            cropped[index] = resizedBytes[scaledIndex];
                            cropped[index + 1] = resizedBytes[scaledIndex + 1];
                            cropped[index + 2] = resizedBytes[scaledIndex + 2];
                            cropped[index + 3] = 255;
                        }
                    }
                    return cropped;
                }
            }
        }

        public byte[] ApplyVignetteToImage(byte[] pixels, int px, int py)
        {
            byte priorr, priorg, priorb;
            double min = Math.Min(px / 2, py / 2);
            double max = Math.Max(px / 2, py / 2);
            double diagonal = Math.Sqrt(min * min + max * max);
            for (int y = 0; y < py; y++)
            {
                for (int x = 0; x < px; x++)
                {
                    double distance = Math.Sqrt((x - px / 2) * (x - px / 2) + (y - py / 2) * (y - py / 2));
                    if (distance > min)
                    {
                        priorb = pixels[4 * x + 4 * y * px];
                        priorg = pixels[4 * x + 4 * y * px + 1];
                        priorr = pixels[4 * x + 4 * y * px + 2];
                        pixels[4 * x + 4 * y * px] = (byte)(priorb * (1 - 0.5 * (distance - min) / (diagonal - min)));
                        pixels[4 * x + 4 * y * px + 1] = (byte)(priorg * (1 - 0.5 * (distance - min) / (diagonal - min)));
                        pixels[4 * x + 4 * y * px + 2] = (byte)(priorr * (1 - 0.5 * (distance - min) / (diagonal - min)));
                    }
                }
            }
            return pixels;
        }

        public async Task SaveImage(SoftwareBitmap softwareBitmap, string fileName)
        {
            // Get the Pictures folder
            StorageFolder picturesFolder = KnownFolders.PicturesLibrary;

            // Create a new file in the Pictures folder
            StorageFile outputFile = await picturesFolder.CreateFileAsync(fileName, CreationCollisionOption.GenerateUniqueName);

            // Open a stream for writing
            using (IRandomAccessStream stream = await outputFile.OpenAsync(FileAccessMode.ReadWrite))
            {
                // Create an encoder with the desired format
                BitmapEncoder encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.PngEncoderId, stream);

                // Set the software bitmap
                encoder.SetSoftwareBitmap(softwareBitmap);

                // Flush the encoder
                await encoder.FlushAsync();
            }
        }

        public async Task<(byte[], int, int)> ReadImage(StorageFile file)
        {
            using (IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read))
            {
                // Create a decoder for the image
                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                // Get the pixel data
                PixelDataProvider pixelData = await decoder.GetPixelDataAsync();
                var pixels = pixelData.DetachPixelData();
                var width = (int)decoder.PixelWidth;
                var height = (int)decoder.PixelHeight;

                // Retrieve the byte array containing the pixel data
                return (pixels, width, height);
            }
        }

        public async Task<byte[]> LoadImage(string fname, int wx, int wy)
        {
            StorageFile file = await StorageFile.GetFileFromPathAsync(fname);

            using (IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read))
            {
                // Create a decoder for the image
                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                // Get the pixel data
                PixelDataProvider pixelData = await decoder.GetPixelDataAsync();
                var pixels = pixelData.DetachPixelData();
                var width = (int)decoder.PixelWidth;
                var height = (int)decoder.PixelHeight;

                (int cropX, int cropY, float ratio) = CalculateCropCoordinates(width, height, wx, wy);

                var image = new byte[4 * wx * wy];
                for (int y = 0; y < wy; y++)
                {
                    for (int x = 0; x < wx; x++)
                    {
                        int pos1 = (int)(cropX + x / ratio) + (int)(cropY + y / ratio) * width;
                        int pos2 = x + y * wx;
                        image[4 * pos2 + 0] = pixels[4 * pos1 + 0];
                        image[4 * pos2 + 1] = pixels[4 * pos1 + 1];
                        image[4 * pos2 + 2] = pixels[4 * pos1 + 2];
                        image[4 * pos2 + 3] = 255;
                    }
                }
                // Retrieve the byte array containing the pixel data
                return image;
            }
        }

        public async Task<(byte[], int, int)> PickImage()
        {
            // Create a file picker
            var openPicker = new FileOpenPicker();

            // Retrieve the window handle (HWND) of the current WinUI 3 window
            var window = App.MainWindow;
            var hWnd = WindowNative.GetWindowHandle(window);

            // Initialize the file picker with the window handle (HWND)
            InitializeWithWindow.Initialize(openPicker, hWnd);

            // Set options for the file picker
            openPicker.ViewMode = PickerViewMode.Thumbnail;
            openPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
            openPicker.FileTypeFilter.Add(".jpg");
            openPicker.FileTypeFilter.Add(".jpeg");
            openPicker.FileTypeFilter.Add(".png");

            // Open the picker for the user to pick a file
            StorageFile file = await openPicker.PickSingleFileAsync();

            if (file != null)
            {
                using (IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read))
                {
                    // Create a decoder for the image
                    BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                    // Get the pixel data
                    PixelDataProvider pixelData = await decoder.GetPixelDataAsync();

                    // Retrieve the byte array containing the pixel data
                    return (pixelData.DetachPixelData(), (int)decoder.PixelWidth, (int)decoder.PixelHeight);
                }
            }
            return (null, 0, 0);
        }


        // ******************
        // Messaging handling
        // ******************
        public void WriteLog(string message)
        {
            for (int l = log.Length - 1; l > 0; l--) { log[l] = log[l - 1]; }
            log[0] = message;
            textLog.Text = "";
            for (int l = 0; l < log.Length; l++) { textLog.Text += log[l] + "\n"; }
        }
    }


    // ***********
    // Thing class
    // ***********
    internal class Thing
    {
        public int id;
        public float X, Y, Z, R, G, B;
        public Vector3 v, a;
        public Vector3 springforce;

        public Thing(int i, float x, float y, float z, float r, float g, float b)
        {
            id = i;
            X = x;
            Y = y;
            Z = z;
            R = r;
            G = g;
            B = b;
        }
    }
}