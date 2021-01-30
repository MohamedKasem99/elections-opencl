#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void shuffle(int *array, int size)
{
    if (size > 1)
    {
        for (int i = 0; i < size - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (size - i) + 1);
            int tmp = array[j];
            array[j] = array[i];
            array[i] = tmp;
        }
    }
}

void write_to_file(char filename[], int *votes, int cands, int voters)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d\n", cands);  //no of candidates
    fprintf(fp, "%d\n", voters); //no of voters

    for (int i = 0; i < voters; i++)
    {
        shuffle(votes, cands);
        for (int j = 0; j < cands; j++)
            fprintf(fp, "%d ", votes[j]);
        fseek(fp, -1, SEEK_CUR);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char *argv[])
{
    int voters, cands;
    if (argc < 2)
    {
        printf("Please enter filename as an argument. Exitting \n");
        exit(1);
    }
    char* filename = argv[1];
    printf("Please enter number of candidates: ");
    scanf("%d", &cands); //number of columns
    printf("Please enter number of voters: ");
    scanf("%d", &voters); //number of rows
    /* Initializes random number generator with different seeds depending on the rank */
    srand(time(0));
    int *votes = (int *)malloc(cands * sizeof(int));
    for (int i = 0; i < cands; i++)
    {
        votes[i] = i + 1;
    }

    // Generate the array of random numbers
    write_to_file(filename, votes, cands, voters);
    free(votes);
    return 0;
}