#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////////////
void getTop2(int *votes, int C, int *top2)
{
    top2[0] = top2[1] = -1;
    int max_vote = INT32_MIN;
    for (int i = 0; i < C; i++)
    {
        if (votes[i] > max_vote)
        {
            top2[0] = i;
            max_vote = votes[i];
        }
    }
    max_vote = INT32_MIN;
    for (int i = 0; i < C; i++)
    {
        if (i == top2[0])
            continue;
        if (votes[i] > max_vote)
        {
            top2[1] = i;
            max_vote = votes[i];
        }
    }
}
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
    fp1 = fopen("input.txt", "r"); // File to read from, should be accessible to all processes
    if (!fp1)
    {
        printf("Couldn't open input.txt\n");
        return 1;
    }
    fscanf(fp1, "%d\n", &candidates); // Read number of Candidates
    fscanf(fp1, "%d\n", &voters);     // Read number of Voters

    int *h_vote_pref = (int *)malloc(sizeof(int) * candidates * voters); // original data set given to device
    int *results = (int *)malloc(sizeof(int) * candidates);              // original data set given to device
    int winner = 0;
    double *percent_votes = (double *)malloc(candidates * sizeof(double)); // Array to hold total votes

    for (int i = 0; i < voters; i++)
    {
        readLine(fp1, candidates, h_vote_pref + i * candidates);
    }

    size_t global; // global domain size for our calculation
    size_t local;  // local domain size for our calculation

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;    // compute device id
    cl_context context;        // compute context
    cl_command_queue commands; // compute command queue
    cl_program program;        // compute program
    cl_kernel round1_kernel;   // compute round1_kernel
    cl_kernel round2_kernel;   // compute round1_kernel

    cl_mem d_vote_pref;

    // round 1 buffers
    cl_mem d_local_vote_sum;
    cl_mem output;

    // round 2 buffers
    cl_mem d_local_top2_sum;
    cl_mem d_top2_sum;
    // Fill our data set with random float values
    //
    char *KernelSource;
    FILE *fp;
    size_t source_size, program_size;

    fp = fopen("red_kernel.cl", "rb");
    if (!fp)
    {
        printf("Failed to load kernels\n");
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
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to bind to platform! \n");
        return EXIT_FAILURE;
    } 
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
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
    round1_kernel = clCreateKernel(program, "round1_k", &err);
    if (!round1_kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute round1_kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    d_vote_pref = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * voters * candidates, NULL, NULL);
    d_local_vote_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * candidates, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * candidates, NULL, NULL);

    d_local_top2_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 2, NULL, NULL);
    d_top2_sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * 2, NULL, NULL);
    if (!d_vote_pref || !output || !d_local_vote_sum || !d_local_top2_sum || !d_top2_sum)
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

    // Set the arguments to our compute round1_kernel
    //
    err = 0;
    err = clSetKernelArg(round1_kernel, 0, sizeof(cl_mem), &d_vote_pref);
    err |= clSetKernelArg(round1_kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(round1_kernel, 2, sizeof(cl_mem), NULL);
    err |= clSetKernelArg(round1_kernel, 3, sizeof(int), &candidates);
    err |= clSetKernelArg(round1_kernel, 4, sizeof(int), &voters);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set round1_kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the round1_kernel on the device
    //
    err = clGetKernelWorkGroupInfo(round1_kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve round1_kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the round1_kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    global = voters * candidates < local ? local : ceil(voters * candidates / (float)local) * local;
    err = clEnqueueNDRangeKernel(commands, round1_kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute round1_kernel!\n");
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
    printf("Round 1 results\n============================\n");
    for (int i = 0; i < candidates; i++)
    {
        percent_votes[i] = results[i] / (double)voters;
        printf("Candidate [%d] got %d/%d which is %0.2lf%%\n", i + 1, results[i], voters, percent_votes[i] * 100);
        if (percent_votes[i] > 0.5)
            winner = i + 1;
    }
    if (winner)
        printf("%d 1\n", winner);
    else
    {
        // Round 2
        int top2[2];
        int round2_sum[2];
        getTop2(results, candidates, top2);
        round2_kernel = clCreateKernel(program, "round2_k", &err);
        if (!round2_kernel || err != CL_SUCCESS)
        {
            printf("Error: Failed to create compute round1_kernel!\n");
            exit(1);
        }
        err = 0;
        err = clSetKernelArg(round2_kernel, 0, sizeof(cl_mem), &d_vote_pref);
        err |= clSetKernelArg(round2_kernel, 1, sizeof(cl_mem), &d_top2_sum);
        err |= clSetKernelArg(round2_kernel, 2, sizeof(cl_mem), NULL);
        err |= clSetKernelArg(round2_kernel, 3, sizeof(int), &candidates);
        err |= clSetKernelArg(round2_kernel, 4, sizeof(int), &voters);
        err |= clSetKernelArg(round2_kernel, 5, sizeof(int), &top2[0]);
        err |= clSetKernelArg(round2_kernel, 6, sizeof(int), &top2[1]);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set round1_kernel arguments! %d\n", err);
            exit(1);
        }
        err = clEnqueueNDRangeKernel(commands, round2_kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err)
        {
            printf("Error: Failed to execute round2_kernel!\n");
            printf("Global: %ld, local: %ld", global, local);
            return EXIT_FAILURE;
        }

        // Wait for the command commands to get serviced before reading back results
        //
        clFinish(commands);

        // Read back the results from the device to verify the output
        //
        err = clEnqueueReadBuffer(commands, d_top2_sum, CL_TRUE, 0, sizeof(int) * 2, round2_sum, 0, NULL, NULL);
        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        printf("Round 2 results\n============================\n");
        int arg_winner = (round2_sum[0] > round2_sum[1]) ? 0 : 1;
        printf("Candidate [%d] got %d/%d which is %0.2lf%%\n", top2[0] + 1, round2_sum[0], voters, (round2_sum[0] / (double)voters) * 100);
        printf("Candidate [%d] got %d/%d which is %0.2lf%%\n", top2[1] + 1, round2_sum[1], voters, (round2_sum[1] / (double)voters) * 100);
        winner = top2[arg_winner] + 1;
        printf("%d 2\n", winner);
        printf("Total time taken %f seconds\n", cpu_time_used);
    }

    // Shutdown and cleanup
    //
    clReleaseMemObject(d_vote_pref);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(round1_kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}