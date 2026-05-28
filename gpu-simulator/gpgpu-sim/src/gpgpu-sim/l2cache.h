// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "../abstract_hardware_model.h"
#include "dram.h"

#include <list>
#include <queue>
#include <vector>

class mem_fetch;

class partition_mf_allocator : public mem_fetch_allocator {
 public:
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;
  }
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    abort();
    return NULL;
  }
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr,
                           unsigned long long cycle) const;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf) const;
 
 private:
  const memory_config *m_memory_config;
};

// Memory partition unit contains all the units assolcated with a single DRAM
// channel.
// - It arbitrates the DRAM channel among multiple sub partitions.
// - It does not connect directly with the interconnection network.
class memory_partition_unit {
 public:
  memory_partition_unit(unsigned partition_id, unsigned chiplet_id, const memory_config *config,
                        class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_partition_unit();

  bool busy() const;

  void cache_cycle(unsigned cycle);
  void dram_cycle();
  void simple_dram_model_cycle();

  void set_done(mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file) const;
  void print_stat(FILE *fp) { m_dram->print_stat(fp); }
  void visualize() const { m_dram->visualize(); }
  void print(FILE *fp) const;
  void get_l2_dram_reorder_stats(unsigned long long &buffered,
                                 unsigned long long &issued,
                                 unsigned long long &forced,
                                 unsigned long long &same_key_issued,
                                 unsigned long long &total_wait,
                                 unsigned long long &max_wait,
                                 unsigned long long &pending) const;
  unsigned dram_queue_length() const;
  unsigned dram_queue_limit() const;
  bool dram_queue_full(bool is_write) const;
  void handle_memcpy_to_gpu(size_t dst_start_addr, unsigned subpart_id,
                            mem_access_sector_mask_t mask);

  class memory_sub_partition *get_sub_partition(int sub_partition_id) {
    return m_sub_partition[sub_partition_id];
  }

  // Power model
  void set_dram_power_stats(unsigned &n_cmd, unsigned &n_activity,
                            unsigned &n_nop, unsigned &n_act, unsigned &n_pre,
                            unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
                            unsigned &n_req) const;

  int global_sub_partition_id_to_local_id(int global_sub_partition_id) const;

  unsigned get_mpid() const { return m_id; }
  unsigned get_chiplet_id() const { return m_chiplet_id; }

  class gpgpu_sim *get_mgpu() const {
    return m_gpu;
  }
  
  void return_credit(int inner_sub_partition_id);
  void erase(mem_fetch *mf, int inner_sub_partition_id);

 private:
  unsigned m_id;
  unsigned m_chiplet_id;
  const memory_config *m_config;
  class memory_stats_t *m_stats;
  class memory_sub_partition **m_sub_partition;
  class dram_t *m_dram;

  class arbitration_metadata {
   public:
    arbitration_metadata(const memory_config *config);

    // check if a subpartition still has credit
    bool has_credits(int inner_sub_partition_id) const;
    // borrow a credit for a subpartition
    void borrow_credit(int inner_sub_partition_id);
    // return a credit from a subpartition
    void return_credit(int inner_sub_partition_id);

    // return the last subpartition that borrowed credit
    int last_borrower() const { return m_last_borrower; }

    void print(FILE *fp) const;

   private:
    // id of the last subpartition that borrowed credit
    int m_last_borrower;

    int m_shared_credit_limit;
    int m_private_credit_limit;

    // credits borrowed by the subpartitions
    std::vector<int> m_private_credit;
    int m_shared_credit;
  };
  arbitration_metadata m_arbitration_metadata;

  // determine wheither a given subpartition can issue to DRAM
  bool can_issue_to_dram(int inner_sub_partition_id);
  bool issue_l2_dram_request(unsigned long long cycle);
  bool issue_l2_dram_request_fifo(unsigned long long cycle);
  bool fill_l2_dram_reorder_buffer(unsigned long long cycle);
  bool issue_l2_dram_reorder_buffer(unsigned long long cycle);
  bool l2_dram_reorder_can_hold(mem_fetch *mf) const;
  bool l2_dram_reorder_same_key(const mem_fetch *a, const mem_fetch *b) const;
  bool l2_dram_reorder_matches_last(const mem_fetch *mf) const;
  void l2_dram_reorder_record_last(const mem_fetch *mf);
  void push_l2_dram_latency_queue(mem_fetch *mf, int spid,
                                  unsigned long long cycle,
                                  bool borrow_credit);

  // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
  struct dram_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::list<dram_delay_t> m_dram_latency_queue;

  struct l2_dram_reorder_entry {
    unsigned long long arrival_cycle;
    int spid;
    mem_fetch *req;
  };
  std::list<l2_dram_reorder_entry> m_l2_dram_reorder_buffer;
  bool m_last_l2_dram_reorder_valid;
  unsigned m_last_l2_dram_reorder_dest;
  unsigned m_last_l2_dram_reorder_dram;
  unsigned m_last_l2_dram_reorder_bank;
  unsigned m_last_l2_dram_reorder_row;
  unsigned m_l2_dram_reorder_run;
  unsigned long long m_l2_dram_reorder_buffered;
  unsigned long long m_l2_dram_reorder_issued;
  unsigned long long m_l2_dram_reorder_forced;
  unsigned long long m_l2_dram_reorder_same_key_issued;
  unsigned long long m_l2_dram_reorder_total_wait;
  unsigned long long m_l2_dram_reorder_max_wait;

  class gpgpu_sim *m_gpu;
};

class memory_sub_partition {
 public:
  memory_sub_partition(unsigned sub_partition_id, unsigned chiplet_id, const memory_config *config,
                       class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_sub_partition();

  unsigned get_id() const { return m_id; }
  unsigned get_chiplet_id() const {return m_chiplet_id; }

  bool busy() const;

  void cache_cycle(unsigned cycle);

  bool full() const;
  bool full(unsigned size) const;
  void push(class mem_fetch *mf, unsigned long long clock_cycle);
  //class mem_fetch *pop();
  //class mem_fetch *top();
  class mem_fetch *local_top();    
  class mem_fetch *local_pop();    
  class mem_fetch *remote_top();    
  class mem_fetch *remote_pop();    

  void set_done(mem_fetch *mf);
  bool remote_request_full() const { return remote_request_q->full(); }
  unsigned remote_request_queue_length() const {
    return remote_request_q->get_n_element();
  }
  unsigned remote_request_queue_capacity() const {
    return remote_request_q->get_max_len();
  }
  void push_remote_request(mem_fetch *mf) {
    remote_request_q->push(mf);
  }
  class mem_fetch *pop_remote_request() const {
    if (!remote_request_q->empty()) {
	mem_fetch* mf = remote_request_q->top();
        remote_request_q->pop();
        return mf;
    } else return NULL;
  }

  //20220820 Shiqing add for SM-side LLC
  void push_remote_reply(mem_fetch *mf) {
    remote_reply_q->push(mf);
  }
  class mem_fetch *pop_remote_reply() const {
    if (!remote_reply_q->empty()) {
	mem_fetch* mf = remote_reply_q->top();
        remote_reply_q->pop();
        return mf;
    } else return NULL;
  }

  unsigned flushL2();
  unsigned invalidateL2();

  // interface to L2_dram_queue
  bool L2_dram_queue_empty() const;
  unsigned L2_dram_queue_length() const { return m_L2_dram_queue->get_n_element(); }
  unsigned L2_dram_queue_capacity() const { return m_L2_dram_queue->get_max_len(); }
  class mem_fetch *L2_dram_queue_top() const;
  void L2_dram_queue_pop();

  // interface to dram_L2_queue
  bool dram_L2_queue_full() const;
  void dram_L2_queue_push(class mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file);
  void print_cache_stat(unsigned &accesses, unsigned &misses) const;
  void print(FILE *fp) const;

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
  void clear_L2cache_stats_pw();

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    m_memcpy_cycle_offset += 1;
  }
  int get_request_turn() {return request_turn; }
  void set_request_turn(int turn) { request_turn = turn; }
  bool L2_icnt_remote_full() {return m_L2_icnt_remote->full();}
  unsigned L2_icnt_remote_length() const { return m_L2_icnt_remote->get_n_element(); }
  unsigned L2_icnt_remote_capacity() const { return m_L2_icnt_remote->get_max_len(); }
  unsigned L2_icnt_local_length() const { return m_L2_icnt_local->get_n_element(); }
  unsigned L2_icnt_local_capacity() const { return m_L2_icnt_local->get_max_len(); }
  unsigned request_tracker_size() const { return m_request_tracker.size(); }
  void L2_icnt_remote_push(mem_fetch *mf) {return m_L2_icnt_remote->push(mf);}

 private:
  // data
  unsigned m_id;  //< the global sub partition ID
  unsigned m_chiplet_id;
  const memory_config *m_config;
  class l2_cache *m_L2cache;
  class L2interface *m_L2interface;
  class gpgpu_sim *m_gpu;
  partition_mf_allocator *m_mf_allocator;

  // model delay of ROP units with a fixed latency
  struct rop_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::queue<rop_delay_t> m_rop;

  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_dram_queue;
  fifo_pipeline<mem_fetch> *m_dram_L2_queue;
  //fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue
  fifo_pipeline<mem_fetch> *remote_request_q;
  int request_turn;
  //20220820 Shiqing add remote_reply_q and reply_turn for SM-side LLC
  fifo_pipeline<mem_fetch> *remote_reply_q;  // reply from a remote LLC after the remote request missing in this local LLC
  int reply_turn;
  fifo_pipeline<mem_fetch> *m_L2_icnt_local;  // L2 cache hit response queue local
  fifo_pipeline<mem_fetch> *m_L2_icnt_remote;  // L2 cache hit response queue remote

  class mem_fetch *L2dramout;
  unsigned long long int wb_addr;

  class memory_stats_t *m_stats;

  std::set<mem_fetch *> m_request_tracker;

  friend class L2interface;

  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to
  // the gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;
};

class L2interface : public mem_fetch_interface {
 public:
  L2interface(memory_sub_partition *unit) { m_unit = unit; }
  virtual ~L2interface() {}
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_L2_dram_queue->full();
  }
  //20220820 Shiqing add bool is_local_dram for SM-side LLC
  virtual bool full(unsigned size, bool write, bool is_local_dram) const {
    // assume read and write packets all same size
    if (is_local_dram) 
      return m_unit->m_L2_dram_queue->full();
    else
      return m_unit->m_L2_icnt_remote->full();
  }
  virtual bool full_inter_port(unsigned size, bool write, unsigned chiplet_id) const { return true;}
  virtual void push(mem_fetch *mf) {
    //20220820 Shiqing 
    //mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
    //m_unit->m_L2_dram_queue->push(mf);
    if (l2_cache_mode == 0) {//memory-side LLC
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
      m_unit->m_L2_dram_queue->push(mf);
    } else if (l2_cache_mode == 1) {//SM-side LLC
      if (mf->get_dest() == m_unit->get_chiplet_id()) { //local access miss
        mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
        m_unit->m_L2_dram_queue->push(mf);
      } else { //remote access miss 
	m_unit->m_L2_icnt_remote->push(mf); //return to icnt
      }
    }
  }

 private:
  memory_sub_partition *m_unit;
};

#endif
