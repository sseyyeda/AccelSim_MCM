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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <list>
#include <set>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../statwrapper.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-sim.h"
#include "histogram.h"
#include "l2cache.h"
#include "l2cache_trace.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"

extern std::map<new_addr_type, new_addr_type*> page_table;

mem_fetch *partition_mf_allocator::alloc(new_addr_type addr,
                                         mem_access_type type, unsigned size,
                                         bool wr,
                                         unsigned long long cycle) const {
  assert(wr);
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, WRITE_PACKET_SIZE, -1, -1, -1,
                                0, 0, true, m_memory_config, cycle);
  printf("ZSQ: ERROR! mf generated in LLC but source_chiplet_id not set\n");
  return mf;
}

mem_fetch *partition_mf_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf =
      new mem_fetch(access, NULL, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE,
                    wid, sid, tpc, 0, 0, true, m_memory_config, cycle, original_mf);

  assert(original_mf != NULL);
  mf->set_source(original_mf->get_source()); 
  mf->set_dest(original_mf->get_dest()); 
  
  return mf;
}
memory_partition_unit::memory_partition_unit(unsigned partition_id, unsigned chiplet_id,
                                             const memory_config *config,
                                             class memory_stats_t *stats,
                                             class gpgpu_sim *gpu)
    : m_id(partition_id),
      m_chiplet_id(chiplet_id),
      m_config(config),
      m_stats(stats),
      m_arbitration_metadata(config),
      m_last_l2_dram_reorder_valid(false),
      m_last_l2_dram_reorder_dest(0),
      m_last_l2_dram_reorder_dram(0),
      m_last_l2_dram_reorder_bank(0),
      m_last_l2_dram_reorder_row(0),
      m_l2_dram_reorder_run(0),
      m_l2_dram_reorder_buffered(0),
      m_l2_dram_reorder_issued(0),
      m_l2_dram_reorder_forced(0),
      m_l2_dram_reorder_same_key_issued(0),
      m_l2_dram_reorder_total_wait(0),
      m_l2_dram_reorder_max_wait(0),
      m_gpu(gpu) {
  m_dram = new dram_t(m_id, chiplet_id, m_config, m_stats, this, gpu);

  m_sub_partition = new memory_sub_partition
      *[m_config->m_n_sub_partition_per_memory_channel];
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    unsigned sub_partition_id =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    m_sub_partition[p] =
        new memory_sub_partition(sub_partition_id, chiplet_id, m_config, stats, gpu);
  }
}

void memory_partition_unit::handle_memcpy_to_gpu(
    size_t addr, unsigned global_subpart_id, mem_access_sector_mask_t mask) {
  //unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
  //std::string mystring = mask.to_string<char, std::string::traits_type,
  //                                      std::string::allocator_type>();
  //MEMPART_DPRINTF(
  //    "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
  //    "global_subpart=%u, sector_mask=%s \n",
  //    addr, p, global_subpart_id, mystring.c_str());
  //TODO
  //m_sub_partition[p]->force_l2_tag_update(
  //    addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
}

memory_partition_unit::~memory_partition_unit() {
  delete m_dram;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    delete m_sub_partition[p];
  }
  delete[] m_sub_partition;
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(
    const memory_config *config)
    : m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
      m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
      m_shared_credit(0) {
  // each sub partition get at least 1 credit for forward progress
  // the rest is shared among with other partitions
  m_private_credit_limit = 1;
  m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size +
                          config->gpgpu_dram_return_queue_size -
                          (config->m_n_sub_partition_per_memory_channel - 1);
  if (config->seperate_write_queue_enabled)
    m_shared_credit_limit += config->gpgpu_frfcfs_dram_write_queue_size;
  if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 or
      config->gpgpu_dram_return_queue_size == 0) {
    m_shared_credit_limit =
        0;  // no limit if either of the queue has no limit in size
  }
  assert(m_shared_credit_limit >= 0);
}

bool memory_partition_unit::arbitration_metadata::has_credits(
    int inner_sub_partition_id) const {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    return true;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    return true;
  } else {
    return false;
  }
}

void memory_partition_unit::arbitration_metadata::borrow_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    m_private_credit[spid] += 1;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    m_shared_credit += 1;
  } else {
    assert(0 && "DRAM arbitration error: Borrowing from depleted credit!");
  }
  m_last_borrower = spid;
}

void memory_partition_unit::arbitration_metadata::return_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] > 0) {
    m_private_credit[spid] -= 1;
  } else {
    m_shared_credit -= 1;
  }
  assert((m_shared_credit >= 0) &&
         "DRAM arbitration error: Returning more than available credits!");
}

void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
  fprintf(fp, "private_credit = ");
  for (unsigned p = 0; p < m_private_credit.size(); p++) {
    fprintf(fp, "%d ", m_private_credit[p]);
  }
  fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
  fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit,
          m_shared_credit_limit);
}

bool memory_partition_unit::busy() const {
  bool busy = !m_l2_dram_reorder_buffer.empty();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    if (m_sub_partition[p]->busy()) {
      busy = true;
    }
  }
  return busy;
}

void memory_partition_unit::cache_cycle(unsigned cycle) {
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->cache_cycle(cycle);
  }
}

void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
  m_dram->visualizer_print(visualizer_file);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->visualizer_print(visualizer_file);
  }
}

// determine whether a given subpartition can issue to DRAM
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
  bool has_dram_resource = m_arbitration_metadata.has_credits(spid);

  MEMPART_DPRINTF(
      "sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
      spid, (sub_partition_contention) ? 'T' : 'F',
      (has_dram_resource) ? 'T' : 'F');

  return (has_dram_resource && !sub_partition_contention);
}

bool memory_partition_unit::l2_dram_reorder_can_hold(mem_fetch *mf) const {
  if (!m_config->gpgpu_l2_dram_reorder_enable) return false;
  if (m_config->gpgpu_l2_dram_reorder_buffer_size == 0) return false;
  if (m_config->gpgpu_l2_dram_reorder_remote_only && mf->get_isLocal())
    return false;
  return true;
}

bool memory_partition_unit::l2_dram_reorder_same_key(const mem_fetch *a,
                                                     const mem_fetch *b) const {
  if (!a || !b) return false;

  switch (m_config->gpgpu_l2_dram_reorder_policy) {
    case 0:
      return false;
    case 1:
      return a->get_dest() == b->get_dest();
    case 2:
      return a->get_dest() == b->get_dest() &&
             a->get_dram_id() == b->get_dram_id();
    case 3:
    default:
      return a->get_dest() == b->get_dest() &&
             a->get_dram_id() == b->get_dram_id() &&
             a->get_tlx_addr().bk == b->get_tlx_addr().bk &&
             a->get_tlx_addr().row == b->get_tlx_addr().row;
  }
}

bool memory_partition_unit::l2_dram_reorder_matches_last(
    const mem_fetch *mf) const {
  if (!m_last_l2_dram_reorder_valid || !mf) return false;

  switch (m_config->gpgpu_l2_dram_reorder_policy) {
    case 0:
      return false;
    case 1:
      return m_last_l2_dram_reorder_dest == mf->get_dest();
    case 2:
      return m_last_l2_dram_reorder_dest == mf->get_dest() &&
             m_last_l2_dram_reorder_dram == mf->get_dram_id();
    case 3:
    default:
      return m_last_l2_dram_reorder_dest == mf->get_dest() &&
             m_last_l2_dram_reorder_dram == mf->get_dram_id() &&
             m_last_l2_dram_reorder_bank == mf->get_tlx_addr().bk &&
             m_last_l2_dram_reorder_row == mf->get_tlx_addr().row;
  }
}

void memory_partition_unit::l2_dram_reorder_record_last(
    const mem_fetch *mf) {
  m_last_l2_dram_reorder_valid = true;
  m_last_l2_dram_reorder_dest = mf->get_dest();
  m_last_l2_dram_reorder_dram = mf->get_dram_id();
  m_last_l2_dram_reorder_bank = mf->get_tlx_addr().bk;
  m_last_l2_dram_reorder_row = mf->get_tlx_addr().row;
}

void memory_partition_unit::push_l2_dram_latency_queue(
    mem_fetch *mf, int spid, unsigned long long cycle, bool borrow_credit) {
  MEMPART_DPRINTF("Issue mem_fetch request %p from sub partition %d to dram\n",
                  mf, spid);

  dram_delay_t d;
  d.req = mf;

  std::map<new_addr_type, new_addr_type *>::iterator iter =
      page_table.find(mf->get_page_addr());
  unsigned long long page_fault_cycles =
      (iter == page_table.end()) ? 0 : (iter->second)[1];

  d.ready_cycle = cycle + m_config->dram_latency + page_fault_cycles;
  m_dram_latency_queue.push_back(d);
  mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, cycle);
  if (borrow_credit) m_arbitration_metadata.borrow_credit(spid);
}

bool memory_partition_unit::issue_l2_dram_request_fifo(
    unsigned long long cycle) {
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) return false;

      m_sub_partition[spid]->L2_dram_queue_pop();
      push_l2_dram_latency_queue(mf, spid, cycle, true);
      return true;
    }
  }
  return false;
}

bool memory_partition_unit::fill_l2_dram_reorder_buffer(
    unsigned long long cycle) {
  bool changed = false;
  bool scanned_all = false;

  while (m_l2_dram_reorder_buffer.size() <
             m_config->gpgpu_l2_dram_reorder_buffer_size &&
         !scanned_all) {
    scanned_all = true;
    int last_issued_partition = m_arbitration_metadata.last_borrower();

    for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
         p++) {
      int spid = (p + last_issued_partition + 1) %
                 m_config->m_n_sub_partition_per_memory_channel;
      if (m_sub_partition[spid]->L2_dram_queue_empty() ||
          !can_issue_to_dram(spid))
        continue;

      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (!l2_dram_reorder_can_hold(mf)) continue;

      m_sub_partition[spid]->L2_dram_queue_pop();
      m_arbitration_metadata.borrow_credit(spid);
      l2_dram_reorder_entry entry;
      entry.arrival_cycle = cycle;
      entry.spid = spid;
      entry.req = mf;
      m_l2_dram_reorder_buffer.push_back(entry);
      m_l2_dram_reorder_buffered++;
      changed = true;
      scanned_all = false;

      if (m_l2_dram_reorder_buffer.size() >=
          m_config->gpgpu_l2_dram_reorder_buffer_size)
        break;
    }
  }

  return changed;
}

bool memory_partition_unit::issue_l2_dram_reorder_buffer(
    unsigned long long cycle) {
  if (m_l2_dram_reorder_buffer.empty()) return false;

  std::list<l2_dram_reorder_entry>::iterator selected =
      m_l2_dram_reorder_buffer.begin();
  bool forced = false;

  if (m_config->gpgpu_l2_dram_reorder_window > 0 &&
      cycle - selected->arrival_cycle >=
          m_config->gpgpu_l2_dram_reorder_window) {
    forced = true;
  } else if (m_config->gpgpu_l2_dram_reorder_policy != 0 &&
             m_l2_dram_reorder_run <
                 m_config->gpgpu_l2_dram_reorder_buffer_size) {
    for (std::list<l2_dram_reorder_entry>::iterator it =
             m_l2_dram_reorder_buffer.begin();
         it != m_l2_dram_reorder_buffer.end(); ++it) {
      if (l2_dram_reorder_matches_last(it->req)) {
        selected = it;
        break;
      }
    }
  }

  mem_fetch *mf = selected->req;
  if (m_dram->full(mf->is_write())) return false;

  unsigned long long wait = cycle - selected->arrival_cycle;
  m_l2_dram_reorder_total_wait += wait;
  if (wait > m_l2_dram_reorder_max_wait) m_l2_dram_reorder_max_wait = wait;
  if (forced) m_l2_dram_reorder_forced++;
  if (l2_dram_reorder_matches_last(mf))
    m_l2_dram_reorder_same_key_issued++;

  int spid = selected->spid;
  bool same_key = l2_dram_reorder_matches_last(mf);
  l2_dram_reorder_record_last(mf);
  m_l2_dram_reorder_run = same_key ? m_l2_dram_reorder_run + 1 : 1;
  m_l2_dram_reorder_issued++;
  m_l2_dram_reorder_buffer.erase(selected);
  push_l2_dram_latency_queue(mf, spid, cycle, false);
  return true;
}

bool memory_partition_unit::issue_l2_dram_request(unsigned long long cycle) {
  if (!m_config->gpgpu_l2_dram_reorder_enable)
    return issue_l2_dram_request_fifo(cycle);

  fill_l2_dram_reorder_buffer(cycle);
  if (issue_l2_dram_reorder_buffer(cycle)) return true;

  return issue_l2_dram_request_fifo(cycle);
}

int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  return (global_sub_partition_id -
          m_id * m_config->m_n_sub_partition_per_memory_channel);
}

void memory_partition_unit::simple_dram_model_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle)) {
    mem_fetch *mf_return = m_dram_latency_queue.front().req;
    if (mf_return->get_access_type() != L1_WRBK_ACC &&
        mf_return->get_access_type() != L2_WRBK_ACC) {
      mf_return->set_reply();

      unsigned dest_global_spid = mf_return->get_sub_partition_id();
      int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
      assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
      if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
        if (mf_return->get_access_type() == L1_WRBK_ACC) {
          m_sub_partition[dest_spid]->set_done(mf_return);
          delete mf_return;
        } else {
          m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
          mf_return->set_status(
              IN_PARTITION_DRAM_TO_L2_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_arbitration_metadata.return_credit(dest_spid);
          MEMPART_DPRINTF(
              "mem_fetch request %p return from dram to sub partition %d\n",
              mf_return, dest_spid);
        }
        m_dram_latency_queue.pop_front();
      }

    } else {
      this->set_done(mf_return);
      delete mf_return;
      m_dram_latency_queue.pop_front();
    }
  }

  issue_l2_dram_request(m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

void memory_partition_unit::dram_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  mem_fetch *mf_return = m_dram->return_queue_top();
  if (mf_return) {
    //20220820 Shiqing modified for SM-side LLC
    //unsigned dest_global_spid = mf_return->get_sub_partition_id();
    unsigned dest_global_spid;
    if (l2_cache_mode == 1 && !mf_return->get_isLocal()) { //SM-side remote reply from dram to remote llc id
      dest_global_spid = mf_return->get_remote_sub_partition_id();  
    } else { //SM-side local reply or mem-side reply from dram to local llc id
      dest_global_spid = mf_return->get_sub_partition_id();  
    }

    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);

    //20220820 SHiqing bypass remote llc in SM-side LLC
    if (m_config->bypass_remote_l2 && (l2_cache_mode == 1) && !mf_return->get_isLocal() && (mf_return->get_dram_id() == m_id) && !m_sub_partition[dest_spid]->L2_icnt_remote_full()) { // SM-side bypass remote llc: remote mf & this partition is its remote partition
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        m_sub_partition[dest_spid]->set_done(mf_return);
        delete mf_return;
      } else {
        m_sub_partition[dest_spid]->L2_icnt_remote_push(mf_return);
        mf_return->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_dram->return_queue_pop();
        m_arbitration_metadata.return_credit(dest_spid);
      }
    } else { /////// start do not bypass remote llc

    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        m_sub_partition[dest_spid]->set_done(mf_return);
        delete mf_return;
      } else {
        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_arbitration_metadata.return_credit(dest_spid);
        MEMPART_DPRINTF(
            "mem_fetch request %p return from dram to sub partition %d\n",
            mf_return, dest_spid);
      }
      m_dram->return_queue_pop();
    }

    }///////end do not bypass remote llc

  } else {
    m_dram->return_queue_pop();
  }

  m_dram->cycle();
  m_dram->dram_log(SAMPLELOG);

  issue_l2_dram_request(m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);

  // DRAM latency queue
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle) &&
      !m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    mem_fetch *mf = m_dram_latency_queue.front().req;
    m_dram_latency_queue.pop_front();
    m_dram->push(mf);
  }
}

void memory_partition_unit::set_done(mem_fetch *mf) { 
  //20220820 Shiqing modified for SM-side LLC
  //unsigned dest_global_spid = mf_return->get_sub_partition_id();
  unsigned global_spid;
  if (l2_cache_mode == 1 && !mf->get_isLocal()) { //SM-side remote reply from dram to remote llc id
    global_spid = mf->get_remote_sub_partition_id();  
  } else { //SM-side local reply or mem-side reply from dram to local llc id
    global_spid = mf->get_sub_partition_id();  
  }
  
  int spid = global_sub_partition_id_to_local_id(global_spid);
  
  assert(m_sub_partition[spid]->get_id() == global_spid);
  if (mf->get_access_type() == L1_WRBK_ACC ||
      mf->get_access_type() == L2_WRBK_ACC) {
    m_arbitration_metadata.return_credit(spid);
    MEMPART_DPRINTF(
        "mem_fetch request %p return from dram to sub partition %d\n", mf,
        spid);
  }
  m_sub_partition[spid]->set_done(mf);
}
  void memory_partition_unit::return_credit(int inner_sub_partition_id) { m_arbitration_metadata.return_credit(inner_sub_partition_id); }
  void memory_partition_unit::erase(mem_fetch *mf, int inner_sub_partition_id) { m_sub_partition[inner_sub_partition_id]->set_done(mf); }

void memory_partition_unit::set_dram_power_stats(
    unsigned &n_cmd, unsigned &n_activity, unsigned &n_nop, unsigned &n_act,
    unsigned &n_pre, unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB, unsigned &n_req) const {
  m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd,
                               n_wr, n_wr_WB, n_req);
}

void memory_partition_unit::get_l2_dram_reorder_stats(
    unsigned long long &buffered, unsigned long long &issued,
    unsigned long long &forced, unsigned long long &same_key_issued,
    unsigned long long &total_wait, unsigned long long &max_wait,
    unsigned long long &pending) const {
  buffered += m_l2_dram_reorder_buffered;
  issued += m_l2_dram_reorder_issued;
  forced += m_l2_dram_reorder_forced;
  same_key_issued += m_l2_dram_reorder_same_key_issued;
  total_wait += m_l2_dram_reorder_total_wait;
  if (m_l2_dram_reorder_max_wait > max_wait)
    max_wait = m_l2_dram_reorder_max_wait;
  pending += m_l2_dram_reorder_buffer.size();
}

unsigned memory_partition_unit::dram_queue_length() const {
  return m_dram->que_length();
}

unsigned memory_partition_unit::dram_queue_limit() const {
  return m_dram->queue_limit();
}

bool memory_partition_unit::dram_queue_full(bool is_write) const {
  return m_dram->full(is_write);
}

void memory_partition_unit::print(FILE *fp) const {
  fprintf(fp, "Memory Partition %u: \n", m_id);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->print(fp);
  }
  fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
          m_dram_latency_queue.size());
  for (std::list<dram_delay_t>::const_iterator mf_dlq =
           m_dram_latency_queue.begin();
       mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
    mem_fetch *mf = mf_dlq->req;
    fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle);
    if (mf)
      mf->print(fp);
    else
      fprintf(fp, " <NULL mem_fetch?>\n");
  }
  if (m_config->gpgpu_l2_dram_reorder_enable) {
    double avg_wait = m_l2_dram_reorder_issued
                          ? (double)m_l2_dram_reorder_total_wait /
                                (double)m_l2_dram_reorder_issued
                          : 0.0;
    fprintf(fp,
            "L2->DRAM reorder: buffered=%llu issued=%llu forced=%llu "
            "same_key_issued=%llu pending=%zu avg_wait=%.2f max_wait=%llu\n",
            m_l2_dram_reorder_buffered, m_l2_dram_reorder_issued,
            m_l2_dram_reorder_forced, m_l2_dram_reorder_same_key_issued,
            m_l2_dram_reorder_buffer.size(), avg_wait,
            m_l2_dram_reorder_max_wait);
  }
  m_dram->print(fp);
}

memory_sub_partition::memory_sub_partition(unsigned sub_partition_id, unsigned chiplet_id,
                                           const memory_config *config,
                                           class memory_stats_t *stats,
                                           class gpgpu_sim *gpu) {
  m_id = sub_partition_id;
  m_chiplet_id = chiplet_id;
  m_config = config;
  m_stats = stats;
  m_gpu = gpu;
  m_memcpy_cycle_offset = 0;

  assert(m_id < m_config->m_n_mem_sub_partition);

  char L2c_name[32];
  snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
  m_L2interface = new L2interface(this);
  m_mf_allocator = new partition_mf_allocator(config);

  if (!m_config->m_L2_config.disabled()) {
    m_L2cache =
        new l2_cache(L2c_name, m_config->m_L2_config, -1, -1, m_L2interface,
                     m_mf_allocator, IN_PARTITION_L2_MISS_QUEUE, gpu);
    m_L2cache->set_chiplet_id(chiplet_id);
  }

  unsigned int icnt_L2;
  unsigned int L2_dram;
  unsigned int dram_L2;
  unsigned int L2_icnt;
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
  m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
  m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
  //m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
  remote_request_q = new fifo_pipeline<mem_fetch>("remote_request", 0, icnt_L2);
  remote_reply_q = new fifo_pipeline<mem_fetch>("remote_reply", 0, dram_L2);
  //std::cout<<"Hossein: remote_request_q size is :"<<icnt_L2<<"\n";
  request_turn = 0;
  reply_turn = 0;
  m_L2_icnt_local = new fifo_pipeline<mem_fetch>("L2-to-icnt-local", 0, L2_icnt);
  m_L2_icnt_remote = new fifo_pipeline<mem_fetch>("L2-to-icnt-remote", 0, L2_icnt);
  wb_addr = -1;
}

memory_sub_partition::~memory_sub_partition() {
  delete m_icnt_L2_queue;
  delete m_L2_dram_queue;
  delete m_dram_L2_queue;
  //delete m_L2_icnt_queue;
  delete remote_request_q;
  delete remote_reply_q;
  delete m_L2_icnt_local;
  delete m_L2_icnt_remote;
  delete m_L2cache;
  delete m_L2interface;
}

void memory_sub_partition::cache_cycle(unsigned cycle) {
  // L2 fill responses
  if (!m_config->m_L2_config.disabled()) {
    if (m_L2cache->access_ready() && !m_L2_icnt_local->full() && !m_L2_icnt_remote->full()) {
      mem_fetch *mf = m_L2cache->next_access();
      if (mf->get_access_type() !=
          L2_WR_ALLOC_R) {  // Don't pass write allocate read request back to
                            // upper level cache
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        //m_L2_icnt_queue->push(mf);
	//20220820 Shiqing add (l2_cache_mode == 0) for memory-side LLC
	if (l2_cache_mode == 0) {
	  if (mf->get_isLocal()) m_L2_icnt_local->push(mf);
          else m_L2_icnt_remote->push(mf);
        } else {//20220820 Shiqing add for SM-side LLC
	  if (!mf->get_isLocal() && (mf->get_dest() == m_chiplet_id)) m_L2_icnt_remote->push(mf);
	  else m_L2_icnt_local->push(mf);
  	}
      } else {
        if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
          mem_fetch *original_wr_mf = mf->get_original_wr_mf();
          assert(original_wr_mf);
          original_wr_mf->set_reply();
          original_wr_mf->set_status(
              IN_PARTITION_L2_TO_ICNT_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          //m_L2_icnt_queue->push(original_wr_mf);
	  //20220820 Shiqing add (l2_cache_mode == 0) for memory-side LLC
	  if (l2_cache_mode == 0) {
	    if (mf->get_isLocal()) m_L2_icnt_local->push(mf);
            else m_L2_icnt_remote->push(mf);
          } else {//20220820 Shiqing add for SM-side LLC
	    if (!mf->get_isLocal() && (mf->get_dest() == m_chiplet_id)) m_L2_icnt_remote->push(original_wr_mf);
	    else m_L2_icnt_local->push(original_wr_mf);
  	  }
        }
        m_request_tracker.erase(mf);
        delete mf;
      }
    }
  }

  //20220820 Shiqing add (l2_cache_mode == 0) for memory-side LLC
  // DRAM to L2 (texture) and icnt (not texture)
  //if (!m_dram_L2_queue->empty()) {
  if (l2_cache_mode == 0 && !m_dram_L2_queue->empty()) {
    mem_fetch *mf = m_dram_L2_queue->top();
    if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
      if (m_L2cache->fill_port_free()) {
        mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset);
        m_dram_L2_queue->pop();
      }
    } else if (!m_L2_icnt_local->full() && !m_L2_icnt_remote->full()) {
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      //m_L2_icnt_queue->push(mf);
      if (mf->get_isLocal()) m_L2_icnt_local->push(mf);
      else m_L2_icnt_remote->push(mf);
      m_dram_L2_queue->pop();
    }
  }
  //20220820 Shiqing add for SM-side LLC
  // DRAM to L2 (texture) and icnt (not texture)
  //if (!m_dram_L2_queue->empty()) {
  if (l2_cache_mode == 1) {
    if (reply_turn == 0) {// get reply from local dram	  
     if (!m_dram_L2_queue->empty()) {
        mem_fetch *mf = m_dram_L2_queue->top();
	if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
          if (m_L2cache->fill_port_free()) {
            mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                           m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
            m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                           m_memcpy_cycle_offset);
	    m_dram_L2_queue->pop();
          }
        } else if (!m_L2_icnt_local->full() && !m_L2_icnt_remote->full()) {
          if (mf->is_write() && mf->get_type() == WRITE_ACK)
            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                           m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
	  if (!mf->get_isLocal() && (mf->get_dest() == m_chiplet_id)) m_L2_icnt_remote->push(mf);
          else m_L2_icnt_local->push(mf);
          m_dram_L2_queue->pop();
        }
 	reply_turn = 1; // next cycle get reply from remote_reply_q	
      } else { // no reply from local dram, get reply from remote_reply_q 
        m_dram_L2_queue->pop();
        mem_fetch *mf = remote_reply_q->top();
	if (mf) {
          if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
            if (m_L2cache->fill_port_free()) {
              mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                             m_memcpy_cycle_offset);
              remote_reply_q->pop();
            }
          } else if (!m_L2_icnt_local->full()) {
            if (mf->is_write() && mf->get_type() == WRITE_ACK)
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
            //if (mf->get_isLocal()) m_L2_icnt_local->push(mf);
            //else m_L2_icnt_remote->push(mf);
            m_L2_icnt_local->push(mf);
            remote_reply_q->pop();
          }
        }
      }
    } else { // get reply from remote_reply_q
        mem_fetch *mf = remote_reply_q->top();
	if (mf) {
          if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
            if (m_L2cache->fill_port_free()) {
              mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                             m_memcpy_cycle_offset);
              remote_reply_q->pop();
            }
          } else if (!m_L2_icnt_local->full()) {
            if (mf->is_write() && mf->get_type() == WRITE_ACK)
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
            //if (mf->get_isLocal()) m_L2_icnt_local->push(mf);
            //else m_L2_icnt_remote->push(mf);
            m_L2_icnt_local->push(mf);
            remote_reply_q->pop();
          }
	  reply_turn = 0; // next cycle get mreply from local dram
        } else { // no reply from remote_reply_q, get reply from local dram
          remote_reply_q->pop();
          if (!m_dram_L2_queue->empty()) {
            mem_fetch *mf = m_dram_L2_queue->top();
            if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
              if (m_L2cache->fill_port_free()) {
                mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                             m_memcpy_cycle_offset);
                m_dram_L2_queue->pop();
              }
            } else if (!m_L2_icnt_local->full() && !m_L2_icnt_remote->full()) {
              if (mf->is_write() && mf->get_type() == WRITE_ACK)
                mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
	      if (!mf->get_isLocal() && (mf->get_dest() == m_chiplet_id)) m_L2_icnt_remote->push(mf);
              else m_L2_icnt_local->push(mf);
              m_dram_L2_queue->pop();
            }
	  }
	} 
    }
  }

  // prior L2 misses inserted into m_L2_dram_queue here
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();

  // new L2 texture accesses and/or non-texture accesses
  if (!m_L2_dram_queue->full() && !m_icnt_L2_queue->empty()) {
    mem_fetch *mf = m_icnt_L2_queue->top();

    //20220820 Shiqing
    if (m_config->bypass_remote_l2 && (l2_cache_mode == 1) && !mf->get_isLocal() && (mf->get_sub_partition_id() != m_id)) { 
      m_L2_dram_queue->push(mf);
      m_icnt_L2_queue->pop();
    } else { /////////// start do not bypass remote llc
   

    if (!m_config->m_L2_config.disabled() &&
        ((m_config->m_L2_texure_only && mf->istexture()) ||
         (!m_config->m_L2_texure_only))) {
      // L2 is enabled and access is for L2
      bool output_full = m_L2_icnt_local->full() || m_L2_icnt_remote->full();
      bool port_free = m_L2cache->data_port_free();
      if (!output_full && port_free) {
        std::list<cache_event> events;
        enum cache_request_status status =
            m_L2cache->access(mf->get_addr(), mf,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                  m_memcpy_cycle_offset,
                              events);
        bool write_sent = was_write_sent(events);
        bool read_sent = was_read_sent(events);
        MEM_SUBPART_DPRINTF("Probing L2 cache Address=%llx, status=%u\n",
                            mf->get_addr(), status);
        if (status == HIT) {
          if (!write_sent) {
            // L2 cache replies
            assert(!read_sent);
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              //m_L2_icnt_queue->push(mf);
	      //20220820 Shiqing add (l2_cache_mode == 0) for memory-side LLC
	      if (l2_cache_mode == 0) {
	        if (mf->get_isLocal()) m_L2_icnt_local->push(mf);
      	        else m_L2_icnt_remote->push(mf);
	      } else {//20220820 Shiqing add for SM-side LLC
	        if (!mf->get_isLocal() && (mf->get_dest() == m_chiplet_id)) m_L2_icnt_remote->push(mf);
		else m_L2_icnt_local->push(mf);
	      }
	    }
   	      m_icnt_L2_queue->pop();
          } else {
            assert(write_sent);
            m_icnt_L2_queue->pop();
          }
        } else if (status != RESERVATION_FAIL) {
          if (mf->is_write() &&
              (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
               m_config->m_L2_config.m_write_alloc_policy ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              //mf->set_reply();
              //mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
              //               m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              //m_L2_icnt_queue->push(mf);
	      //20220820 Shiqing add (l2_cache_mode == 0) for memory-side LLC
	      if (l2_cache_mode == 0) {
                mf->set_reply();
                mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                               m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
	        if (mf->get_isLocal()) m_L2_icnt_local->push(mf);
                else m_L2_icnt_remote->push(mf);
              } else {//20220820 Shiqing add for SM-side LLC
	        if (!mf->get_isLocal() && (mf->get_dest() == m_chiplet_id)) m_L2_icnt_remote->push(mf);
		else {
                  mf->set_reply();
                  mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                               m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
		  m_L2_icnt_local->push(mf);
		}
  	      }
            }
          }
          // L2 cache accepted request
          m_icnt_L2_queue->pop();
        } else {
          assert(!write_sent);
          assert(!read_sent);
          // L2 cache lock-up: will try again next cycle
        }
      }
    } else {
      // L2 is disabled or non-texture access to texture-only L2
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_dram_queue->push(mf);
      m_icnt_L2_queue->pop();
    }

    } /////////////end do not bypass remote llc

  }

  // ROP delay queue
  if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
      !m_icnt_L2_queue->full()) {
    mem_fetch *mf = m_rop.front().req;
    m_rop.pop();
    m_icnt_L2_queue->push(mf);
    mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  }
}

bool memory_sub_partition::full() const { return m_icnt_L2_queue->full(); }

bool memory_sub_partition::full(unsigned size) const {
  return m_icnt_L2_queue->is_avilable_size(size);
}

bool memory_sub_partition::L2_dram_queue_empty() const {
  return m_L2_dram_queue->empty();
}

class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
  return m_L2_dram_queue->top();
}

void memory_sub_partition::L2_dram_queue_pop() { m_L2_dram_queue->pop(); }

bool memory_sub_partition::dram_L2_queue_full() const {
  return m_dram_L2_queue->full();
}

void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
  m_dram_L2_queue->push(mf);
}

void memory_sub_partition::print_cache_stat(unsigned &accesses,
                                            unsigned &misses) const {
  FILE *fp = stdout;
  if (!m_config->m_L2_config.disabled()) m_L2cache->print(fp, accesses, misses);
}

void memory_sub_partition::print(FILE *fp) const {
  if (!m_request_tracker.empty()) {
    fprintf(fp, "Memory Sub Parition %u: pending memory requests:\n", m_id);
    for (std::set<mem_fetch *>::const_iterator r = m_request_tracker.begin();
         r != m_request_tracker.end(); ++r) {
      mem_fetch *mf = *r;
      if (mf)
        mf->print(fp);
      else
        fprintf(fp, " <NULL mem_fetch?>\n");
    }
  }
  if (!m_config->m_L2_config.disabled()) m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print(gzFile visualizer_file) {
  gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
  gzprintf(visualizer_file, "Ltwowritehit: %d\n", L2_write_hit);
  gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
  gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
  clear_L2_stats_pw();

  if (num_mfs)
    gzprintf(visualizer_file, "averagemflatency: %lld\n",
             mf_total_lat / num_mfs);
}

void memory_stats_t::clear_L2_stats_pw() {
  L2_write_miss = 0;
  L2_write_hit = 0;
  L2_read_miss = 0;
  L2_read_hit = 0;
}

void gpgpu_sim::print_dram_stats(FILE *fout) const {
  unsigned cmd = 0;
  unsigned activity = 0;
  unsigned nop = 0;
  unsigned act = 0;
  unsigned pre = 0;
  unsigned rd = 0;
  unsigned wr = 0;
  unsigned wr_WB = 0;
  unsigned req = 0;
  unsigned tot_cmd = 0;
  unsigned tot_nop = 0;
  unsigned tot_act = 0;
  unsigned tot_pre = 0;
  unsigned tot_rd = 0;
  unsigned tot_wr = 0;
  unsigned tot_req = 0;

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i]->set_dram_power_stats(cmd, activity, nop, act,
                                                     pre, rd, wr, wr_WB, req);
    tot_cmd += cmd;
    tot_nop += nop;
    tot_act += act;
    tot_pre += pre;
    tot_rd += rd;
    tot_wr += wr + wr_WB;
    tot_req += req;
  }
  fprintf(fout, "gpgpu_n_dram_reads = %d\n", tot_rd);
  fprintf(fout, "gpgpu_n_dram_writes = %d\n", tot_wr);
  fprintf(fout, "gpgpu_n_dram_activate = %d\n", tot_act);
  fprintf(fout, "gpgpu_n_dram_commands = %d\n", tot_cmd);
  fprintf(fout, "gpgpu_n_dram_noops = %d\n", tot_nop);
  fprintf(fout, "gpgpu_n_dram_precharges = %d\n", tot_pre);
  fprintf(fout, "gpgpu_n_dram_requests = %d\n", tot_req);
}

unsigned memory_sub_partition::flushL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->flush();
  }
  return 0;  // TODO: write the flushed data to the main memory
}

unsigned memory_sub_partition::invalidateL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->invalidate();
  }
  return 0;
}

bool memory_sub_partition::busy() const { return !m_request_tracker.empty(); }

std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    result.push_back(mf);
  } else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    // break down every sector
    mem_access_byte_mask_t mask;
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf);

      result.push_back(n_mf);
    }
    // This is for constant cache
  } else if (mf->get_data_size() == 64 &&
             (mf->get_access_sector_mask().all() ||
              mf->get_access_sector_mask().none())) {
    unsigned start;
    if (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE == 0)
      start = 0;
    else
      start = 2;
    mem_access_byte_mask_t mask;
    for (unsigned i = start; i < start + 2; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr(), mf->get_access_type(), mf->get_access_warp_mask(),
          mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf);

      result.push_back(n_mf);
    }
  } else {
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k);
        }
        mem_fetch *n_mf = m_mf_allocator->alloc(
            mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf);

        result.push_back(n_mf);
      }
    }
  }
  if (result.size() == 0) assert(0 && "no mf sent");
  return result;
}

void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    m_stats->memlatstat_icnt2mem_pop(m_req);
    std::vector<mem_fetch *> reqs;
    if (m_config->m_L2_config.m_cache_type == SECTOR)
      reqs = breakdown_request_to_sector_requests(m_req);
    else
      reqs.push_back(m_req);

    for (unsigned i = 0; i < reqs.size(); ++i) {
      mem_fetch *req = reqs[i];
      m_request_tracker.insert(req);
      if (req->istexture()) {
        m_icnt_L2_queue->push(req);
        req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      } else {
        rop_delay_t r;
        r.req = req;
        r.ready_cycle = cycle + m_config->rop_latency;
        m_rop.push(r);
        req->set_status(IN_PARTITION_ROP_DELAY,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      }
    }
  }
}

/* gpgpusim-v4
mem_fetch *memory_sub_partition::pop() {
  mem_fetch *mf = m_L2_icnt_queue->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}

mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}
*/

//# Shiqing Zhang: multi-chiplet implementation below
mem_fetch *memory_sub_partition::local_top() {
  mem_fetch *mf = m_L2_icnt_local->top();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_local->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}
mem_fetch *memory_sub_partition::remote_top() {
  mem_fetch *mf = m_L2_icnt_remote->top();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_remote->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}

mem_fetch *memory_sub_partition::local_pop() {
  mem_fetch *mf = m_L2_icnt_local->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}
mem_fetch *memory_sub_partition::remote_pop() {
  mem_fetch *mf = m_L2_icnt_remote->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}

void memory_sub_partition::set_done(mem_fetch *mf) {   
  m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(
    class cache_stats &l2_stats) const {
  if (!m_config->m_L2_config.disabled()) {
    l2_stats += m_L2cache->get_stats();
  }
}

void memory_sub_partition::get_L2cache_sub_stats(
    struct cache_sub_stats &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats(css);
  }
}

void memory_sub_partition::get_L2cache_sub_stats_pw(
    struct cache_sub_stats_pw &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats_pw(css);
  }
}

void memory_sub_partition::clear_L2cache_stats_pw() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->clear_pw();
  }
}

void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
  // Support for L2 AerialVision stats
  // Per-sub-partition stats would be trivial to extend from this
  cache_sub_stats_pw temp_sub_stats;
  get_L2cache_sub_stats_pw(temp_sub_stats);

  m_stats->L2_read_miss += temp_sub_stats.read_misses;
  m_stats->L2_write_miss += temp_sub_stats.write_misses;
  m_stats->L2_read_hit += temp_sub_stats.read_hits;
  m_stats->L2_write_hit += temp_sub_stats.write_hits;

  clear_L2cache_stats_pw();
}
