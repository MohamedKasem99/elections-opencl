kernel void round1_k(__global int *vote_pref, __global int *vote_summary,
                     __local int *local_vote_sum, const int candidates,
                     const int voters) {
  int idx = get_global_id(0);
  if (idx < voters) {
    int cand_choice = vote_pref[idx * candidates];
    for (int i = 0; i < candidates; i++) {
      local_vote_sum[i] = work_group_reduce_add((i + 1) == cand_choice ? 1 : 0);
    }
    if (idx == get_group_id(0) * get_local_size(0)) {
      for (int i = 0; i < candidates; i++)
        atomic_add(&vote_summary[i], local_vote_sum[i]);
    }
  }
}

kernel void round2_k(__global int *vote_pref, __global int *top2_sum,
                     __local int *local_top2_sum, const int candidates,
                     const int voters, const int cand_1, const int cand_2) {
  int idx = get_global_id(0);
  int tmp_choice;
  int cand_choice = 0;
  if (idx < voters) {

    for (int i = 0; i < candidates; i++) {
      tmp_choice = vote_pref[idx * candidates + i] - 1;
      if (tmp_choice == cand_1 || tmp_choice == cand_2){
        cand_choice = tmp_choice;
        break;
      }
    }

    local_top2_sum[0] = work_group_reduce_add( cand_1 == cand_choice? 1 : 0);
    local_top2_sum[1] = work_group_reduce_add( cand_2 == cand_choice? 1 : 0);

    if (idx == get_group_id(0) * get_local_size(0)) {
        atomic_add(&top2_sum[0], local_top2_sum[0]);
        atomic_add(&top2_sum[1], local_top2_sum[1]);
    }
  }
}