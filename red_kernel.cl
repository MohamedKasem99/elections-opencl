kernel void vote_sum(__global int *vote_pref, __global int *vote_summary,
                     const uint candidates, const uint voters) {
  int idx = get_global_id(0);
  if (idx < voters) {
    // int local_size = get_local_size(0);
    // int global_size = get_global_size(0);
    // int groupId = get_group_id(0);
    int cand_choice = vote_pref[idx * candidates];
    for (int i = 0; i < candidates; i++) {
      vote_summary[i] = work_group_reduce_add((i + 1) == cand_choice ? 1 : 0);
    }
  }
}