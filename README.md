# elections-opencl
Distributed election program using opencl

# Team:
* Mohamed Kasem 201601144
* Mohamed Mostafa 201600236

## Task 1 input generation
* First, at rank zero, the user is prompted to enter the number of candidates and voters.
* Then, create an array called `votes`, which has the values from 1 to number of candidates.
* `shuffle` function shuffles the order of values. This ensures that a single voter would never have repeated candidate numbers in his/her votes.
* To ensure that shuffle creates different values for each process, and different values everytime a user runs the program, we used `srand(time(0))`
* The results are then serially written into a file.

## Task 2 result calculation
1. Preperations before execution
    * The kernels are written in another file called `red_kernel.cl` and read into the host program.
    * The program uses the first gpu available to it.
    * The workgroups are set to the maximum size it can be.
    * The number of global work items are set to be the nearest multiple of local work item size in a workgroup so that each work item deals with one voter.
    * The host intializes memory to hold the votes from the input file which is allocated to the global memory for all workgroups to access.
    * A result array is intialized in the global memory.
    * A local result array is initialized in the local memory of each workgroup.
2. Read the file
    * The file is read into the global memory where all workgroups can access it.
3. First round
    * Each work item gets its voter's first pick
    * Increments the local result array using `work_group_reduce_add`, so that each workgroup has a local array that holds the workgroup's total votes for each candidate.
    * All workgroups then atomically add the result from the local result array into the global result array.
    * The percentages is then calculated at the host.
    * If any candidate gets more than 50% then we have a winner
    * Otherwise we move on to Round 2
3. Second round
    * Call the round 2 kernel with the top 2 candidates from round 1.
    * The same reduction in round 1 occurs but with a result array of size 2 instead.
    * The votes are converted to percentages and printed to `stdout`
    * Get the candidate with score >= 50% and print it as winner.
