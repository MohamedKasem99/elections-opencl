
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

void readLine(FILE *fp, int C, int *out)
{
    // Reads one voter's votes line
    for (int i = 0; i < C - 1; i++)
    {
        fscanf(fp, "%d ", &out[i]);
    }
    fscanf(fp, "%d\n", &out[C - 1]);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    int err; // error code returned from api calls

    int candidates;
    int voters;
    FILE *fp1;
    fp1 = fopen("input.txt", "r");                                   // File to read from, should be accessible to all processes
    if (!fp1)
    {
        printf("Couldn't open input.txt\n");
        return 1;
    }
    fscanf(fp1, "%d\n", &candidates);                                     // Read number of Candidates
    fscanf(fp1, "%d\n", &voters);                                         // Read number of Voters
    int *h_vote_pref = (int *)malloc(sizeof(int) * candidates * voters); // original data set given to device
    int *results = (int *)malloc(sizeof(int) * candidates);              // original data set given to device
    unsigned int correct;                                                // number of correct results returned

    for (int i = 0; i < voters; i++)
    {
        readLine(fp1, candidates, h_vote_pref + i * candidates);
    }

    size_t global; // global domain size for our calculation
    size_t local;  // local domain size for our calculation

    cl_device_id device_id;    // compute device id
    cl_context context;        // compute context
    cl_command_queue commands; // compute command queue
    cl_program program;        // compute program
    cl_kernel kernel;          // compute kernel

    cl_mem d_vote_pref; // device memory used for the input array
    cl_mem output;      // device memory used for the output array

    // Fill our data set with random float values
    //
    char *KernelSource;
    FILE *fp;
    size_t source_size, program_size;

    fp = fopen("red_kernel.cl", "rb");
    if (!fp)
    {
        printf("Failed to load kernel\n");
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);
    KernelSource = (char *)malloc(program_size + 1);
    KernelSource[program_size] = '\0';
    fread(KernelSource, sizeof(char), program_size, fp);
    fclose(fp);

    // Connect to a compute device
    //
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "vote_sum", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    d_vote_pref = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * voters * candidates, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * candidates, NULL, NULL);
    if (!d_vote_pref || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands, d_vote_pref, CL_TRUE, 0, sizeof(int) * voters * candidates, h_vote_pref, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vote_pref);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &candidates);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &voters);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = voters * candidates < local ? local : ceil(voters * candidates / (float)local) * local;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        printf("Global: %ld, local: %ld", global, local);
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(int) * candidates, results, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    // Validate our results
    //
    correct = 0;
    printf("results: ");
    for (int i = 0; i < candidates; i++)
    {
        printf("%d ", results[i]);
    }
    printf("\n");

    // Shutdown and cleanup
    //
    clReleaseMemObject(d_vote_pref);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}