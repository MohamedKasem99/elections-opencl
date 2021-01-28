# elections-opencl
Simple tutorial on how to use opencl to simulate an election process.

#### Project Statement
The results of any elections may take weeks to be announced. But we want to quickly announce
which candidate will win. So given the voters’ preferences lists, you need to write a parallel
program to announce which candidate will win and in which round.

#### Details

When it is time to vote for a new president and as a voter you are really excited about that. You
know that the final results may take weeks to be announced, while you can't really wait to see the
results. Suppose the preferences list is available for every voter. Each voter sorted out all
candidates starting by his most preferred candidate and ending with his least preferred one.
When voting, a voter votes for the candidate who comes first in his preferences list. For example,
if there are 5 candidates (numbered 1 to 5), and the preferences list for one voter is [3, 2, 5, 1, 4]
then voter will give the highest vote for candidate 3 and the lowest vote for candidate 4.
**The rules for the election process are as follows:**

1. There are C candidates (numbered from 1 to C), and V voters.
2. The election process consists of up to 2 rounds. All candidates compete in the first round. If a
candidate receives more than 50% of the votes, he wins, otherwise another round takes place,
in which only the top 2 candidates compete for the presidency, the candidate who receives
more votes than his opponent wins and becomes the new president.
3. The voters' preferences are the same in both rounds so if the preference list [1 2 3 4 5] in the
first round and the second round become between candidate 1 and 2 so the preferences is the
same [1 2].
Given the preferences lists, you need to write a parallel program to announce which candidate
will win and in which round.

**For example:** If the input is:
3 5 // number of candidates & number of voters
1 2 3 // voter 1 preference list
1 2 3 // voter 2 preference list
2 1 3 // voter 3 preference list
2 3 1 // voter 4 preference list
3 2 1 // voter 5 preference list
Then the output will be 2 2 // candidate 2 wins in round 2

**Explanation**: You should print the output something like this:
Candidate [1] got 2/5 which is 40%
Candidate [2] got 2/5 which is 40%
Candidate [3] got 1/5 which is 20%
So second round will take place between candidates 1 and 2 with same preferences
1 2 // voter 1 preference list
1 2 // voter 2 preference list
2 1 // voter 3 preference list
2 1 // voter 4 preference list
2 1 // voter 5 preference list
Candidate [1] got 2/5 which is 40%
Candidate [2] got 3/5 which is 60% so candidate 2 wins in round 2