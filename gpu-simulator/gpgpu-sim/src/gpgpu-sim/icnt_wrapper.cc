// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#include "icnt_wrapper.h"
#include <assert.h>
#include "../intersim2/globals.hpp"
#include "../intersim2/interconnect_interface.hpp"
#include "local_interconnect.h"

icnt_create_p icnt_create;
icnt_init_p icnt_init;
icnt_has_buffer_p icnt_has_buffer;
icnt_push_p icnt_push;
icnt_pop_p icnt_pop;
icnt_transfer_p icnt_transfer;
icnt_busy_p icnt_busy;
icnt_display_stats_p icnt_display_stats;
icnt_display_overall_stats_p icnt_display_overall_stats;
icnt_display_state_p icnt_display_state;
icnt_get_flit_size_p icnt_get_flit_size;

/*
//Shiqing Zhang: for compiling and testing
icnt_create_p icnt_create_chLet;
icnt_init_p icnt_init_chLet;
icnt_has_buffer_p icnt_has_buffer_chLet;
icnt_push_p icnt_push_chLet;
icnt_pop_p icnt_pop_chLet;
icnt_transfer_p icnt_transfer_chLet;
icnt_busy_p icnt_busy_chLet;
*/

//Added by s.h.seyyedaghaei@gmail.com. Icnt_wrapper.cc
icnt_create_p icnt_create_chLet;
icnt_init_p icnt_init_chLet;
icnt_has_buffer_p icnt_has_buffer_chLet;
icnt_push_p icnt_push_chLet;
icnt_pop_p icnt_pop_chLet;
icnt_transfer_p icnt_transfer_chLet;
icnt_busy_p icnt_busy_chLet;
icnt_display_stats_p icnt_display_stats_chLet;
icnt_display_overall_stats_p icnt_display_overall_stats_chLet;
icnt_display_state_p icnt_display_state_chLet;
icnt_get_flit_size_p icnt_get_flit_size_chLet;
//net_id ==1 --> local network
//net_id ==2 --> remote network
int net_id;
//End s.h.seyyedaghaei

unsigned g_network_mode;
char* g_network_config_filename;
char* g_network_config_filename_chLet;
struct inct_config g_inct_config;
LocalInterconnect* g_localicnt_interface;

#include "../option_parser.h"

// Wrapper to intersim2 to accompany old icnt_wrapper
// TODO: use delegate/boost/c++11<funtion> instead

static void intersim2_create(unsigned int n_shader, unsigned int n_mem) {
    net_id = 1;
  g_icnt_interface->CreateInterconnect(n_shader, n_mem,net_id);
}

static void intersim2_init() { g_icnt_interface->Init(); }

static bool intersim2_has_buffer(unsigned input, unsigned int size,int subnetId) {
  return g_icnt_interface->HasBuffer(input, size,subnetId);
}

static void intersim2_push(unsigned input, unsigned output, void* data,
                           unsigned int size , int subnetId) {
	g_icnt_interface->Push(input, output, data, size,subnetId);
}

static void* intersim2_pop(unsigned output,int subnetId) {
 
  return g_icnt_interface->Pop(output,subnetId);
}

static void intersim2_transfer() {
    net_id = 1;
    g_icnt_interface->Advance(net_id); }

static bool intersim2_busy() { return g_icnt_interface->Busy(); }

static void intersim2_display_stats() { g_icnt_interface->DisplayStats(); }

static void intersim2_display_overall_stats() {
  g_icnt_interface->DisplayOverallStats();
}

static void intersim2_display_state(FILE* fp) {
  g_icnt_interface->DisplayState(fp);
}

static unsigned intersim2_get_flit_size() {
  return g_icnt_interface->GetFlitSize();
}


//Added by s.h.seyyedaghaei@gmail.com. Add Second Network 
static void intersim2_create_chLet(unsigned int n_chipLet_firstPart, unsigned int n_chiplet_secondPart) {
  net_id = 2;
  g_icnt_interface_chLet->CreateInterconnect(n_chipLet_firstPart, n_chiplet_secondPart,net_id);
}

static void intersim2_init_chLet() { 
  g_icnt_interface_chLet->Init(); 
}

static bool intersim2_has_buffer_chLet(unsigned input, unsigned int size,int subnetId) {
  return g_icnt_interface_chLet->HasBuffer(input, size,subnetId);
}

static void intersim2_push_chLet(unsigned input, unsigned output, void* data,
                           unsigned int size,int subnetId) {
  
  g_icnt_interface_chLet->Push(input, output, data, size,subnetId);
}

static void* intersim2_pop_chLet(unsigned output,int subnetId) {
  return g_icnt_interface_chLet->Pop(output,subnetId);
}

static void intersim2_transfer_chLet() {
    net_id = 2;
    g_icnt_interface_chLet->Advance(net_id); }

static bool intersim2_busy_chLet() { return g_icnt_interface_chLet->Busy(); }

static void intersim2_display_stats_chLet() { g_icnt_interface_chLet->DisplayStats(); }

static void intersim2_display_overall_stats_chLet() {
  g_icnt_interface_chLet->DisplayOverallStats();
}

static void intersim2_display_state_chLet(FILE* fp) {
  g_icnt_interface_chLet->DisplayState(fp);
}

static unsigned intersim2_get_flit_size_chLet() {
  return g_icnt_interface_chLet->GetFlitSize();
}
//End added by s.h.seyyedaghaei
//////////////////////////////////////////////////////

static void LocalInterconnect_create(unsigned int n_shader,
                                     unsigned int n_mem) {
  g_localicnt_interface->CreateInterconnect(n_shader, n_mem);
}

static void LocalInterconnect_init() { g_localicnt_interface->Init(); }

static bool LocalInterconnect_has_buffer(unsigned input, unsigned int size,int subnetId) {
  return g_localicnt_interface->HasBuffer(input, size);
}

static void LocalInterconnect_push(unsigned input, unsigned output, void* data,
                                   unsigned int size,int subnetId) {
  g_localicnt_interface->Push(input, output, data, size);
}

static void* LocalInterconnect_pop(unsigned output,int subnetId) {
  return g_localicnt_interface->Pop(output);
}

static void LocalInterconnect_transfer() { g_localicnt_interface->Advance(); }

static bool LocalInterconnect_busy() { return g_localicnt_interface->Busy(); }

static void LocalInterconnect_display_stats() {
  g_localicnt_interface->DisplayStats();
}

static void LocalInterconnect_display_overall_stats() {
  g_localicnt_interface->DisplayOverallStats();
}

static void LocalInterconnect_display_state(FILE* fp) {
  g_localicnt_interface->DisplayState(fp);
}

static unsigned LocalInterconnect_get_flit_size() {
  return g_localicnt_interface->GetFlitSize();
}

///////////////////////////

void icnt_reg_options(class OptionParser* opp) {
  option_parser_register(opp, "-network_mode", OPT_INT32, &g_network_mode,
                         "Interconnection network mode", "1");
  option_parser_register(opp, "-inter_config_file", OPT_CSTR,
                         &g_network_config_filename,
                         "Interconnection network config file", "mesh");
   option_parser_register(opp, "-inter_config_file_chLet", OPT_CSTR,
                       &g_network_config_filename_chLet,
                        "Interconnection network config file", "inter_config_file_chLet");
  // parameters for local xbar
  option_parser_register(opp, "-icnt_in_buffer_limit", OPT_UINT32,
                         &g_inct_config.in_buffer_limit, "in_buffer_limit",
                         "64");
  option_parser_register(opp, "-icnt_out_buffer_limit", OPT_UINT32,
                         &g_inct_config.out_buffer_limit, "out_buffer_limit",
                         "64");
  option_parser_register(opp, "-icnt_subnets", OPT_UINT32,
                         &g_inct_config.subnets, "subnets", "2");
  option_parser_register(opp, "-icnt_arbiter_algo", OPT_UINT32,
                         &g_inct_config.arbiter_algo, "arbiter_algo", "1");
  option_parser_register(opp, "-icnt_verbose", OPT_UINT32,
                         &g_inct_config.verbose, "inct_verbose", "0");
  option_parser_register(opp, "-icnt_grant_cycles", OPT_UINT32,
                         &g_inct_config.grant_cycles, "grant_cycles", "1");
}

void icnt_wrapper_init() {
  switch (g_network_mode) {
    case INTERSIM:
      // FIXME: delete the object: may add icnt_done wrapper
      net_id = 1;
      g_icnt_interface = InterconnectInterface::New(g_network_config_filename,net_id);
      icnt_create = intersim2_create;
      icnt_init = intersim2_init;
      icnt_has_buffer = intersim2_has_buffer;
      icnt_push = intersim2_push;
      icnt_pop = intersim2_pop;
      icnt_transfer = intersim2_transfer;
      icnt_busy = intersim2_busy;
      icnt_display_stats = intersim2_display_stats;
      icnt_display_overall_stats = intersim2_display_overall_stats;
      icnt_display_state = intersim2_display_state;
      icnt_get_flit_size = intersim2_get_flit_size;

      break;
    case LOCAL_XBAR:
      g_localicnt_interface = LocalInterconnect::New(g_inct_config);
      icnt_create = LocalInterconnect_create;
      icnt_init = LocalInterconnect_init;
      icnt_has_buffer = LocalInterconnect_has_buffer;
      icnt_push = LocalInterconnect_push;
      icnt_pop = LocalInterconnect_pop;
      icnt_transfer = LocalInterconnect_transfer;
      icnt_busy = LocalInterconnect_busy;
      icnt_display_stats = LocalInterconnect_display_stats;
      icnt_display_overall_stats = LocalInterconnect_display_overall_stats;
      icnt_display_state = LocalInterconnect_display_state;
      icnt_get_flit_size = LocalInterconnect_get_flit_size;
/*
      //Shiqing Zhang: for compiling and testing
      icnt_create_chLet = LocalInterconnect_create;
      icnt_init_chLet = LocalInterconnect_init;
      icnt_has_buffer_chLet = LocalInterconnect_has_buffer;
      icnt_push_chLet = LocalInterconnect_push;
      icnt_pop_chLet = LocalInterconnect_pop;
      icnt_transfer_chLet = LocalInterconnect_transfer;
      icnt_busy_chLet = LocalInterconnect_busy;
*/
      break;
    default:
      assert(0);
      break;
  }
}  
//Added by s.h.seyyedaghaei@gmail.com. Implement icnt_wrapper_init_chLet
void icnt_wrapper_init_chLet()
{
    switch (g_network_mode) {
    case INTERSIM:
      // FIXME: delete the object: may add icnt_done wrapper
      net_id = 2;
      g_icnt_interface_chLet = InterconnectInterface::New(g_network_config_filename_chLet,net_id);
      icnt_create_chLet = intersim2_create_chLet;
      icnt_init_chLet = intersim2_init_chLet;
      icnt_has_buffer_chLet = intersim2_has_buffer_chLet;
      icnt_push_chLet = intersim2_push_chLet;
      icnt_pop_chLet = intersim2_pop_chLet;
      icnt_transfer_chLet = intersim2_transfer_chLet;
      icnt_busy_chLet = intersim2_busy_chLet;
      icnt_display_stats_chLet = intersim2_display_stats_chLet;
      icnt_display_overall_stats_chLet = intersim2_display_overall_stats_chLet;
      icnt_display_state_chLet = intersim2_display_state_chLet;
      icnt_get_flit_size_chLet = intersim2_get_flit_size_chLet;
      break;
     case LOCAL_XBAR:
      g_localicnt_interface = LocalInterconnect::New(g_inct_config);
      icnt_create_chLet = LocalInterconnect_create;
      icnt_init_chLet = LocalInterconnect_init;
      icnt_has_buffer_chLet = LocalInterconnect_has_buffer;
      icnt_push_chLet = LocalInterconnect_push;
      icnt_pop_chLet = LocalInterconnect_pop;
      icnt_transfer_chLet = LocalInterconnect_transfer;
      icnt_busy_chLet = LocalInterconnect_busy;
      icnt_display_stats_chLet = LocalInterconnect_display_stats;
      icnt_display_overall_stats_chLet = LocalInterconnect_display_overall_stats;
      icnt_display_state_chLet = LocalInterconnect_display_state;
      icnt_get_flit_size_chLet = LocalInterconnect_get_flit_size;
      break;
    default:
      assert(0);
      break;
    }
}
//End s.h.seyyedaghaei@gmail.com

