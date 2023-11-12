// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
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

#include "scoreboard.h"
#include "../cuda-sim/ptx_sim.h"
#include "shader.h"
#include "shader_trace.h"
#include <iostream>
#include <random>

// Constructor
Scoreboard::Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t* gpu)
    : longopregs() {
  m_sid = sid;
  // Initialize size of table
  reg_table.resize(n_warps);
  reg_table_WAR.resize(n_warps);
  addr_table.resize(n_warps);
  longopregs.resize(n_warps);

  /* New variables for FAST */
  reg_table_mem.resize(n_warps);
  reg_table_comp.resize(n_warps);

  reg_reserved_mem.resize(n_warps);
  reg_pc.resize(n_warps);
  reg_type_mem.resize(n_warps);
  reg_load_type.resize(n_warps);
  reg_released_mem.resize(n_warps);
  reg_reserved_comp.resize(n_warps);
  reg_released_comp.resize(n_warps);
  reg_reserved.resize(n_warps);
  reg_released.resize(n_warps);
  warp_inst_issue_num.resize(n_warps);
  reg_table_all_regs_used.resize(n_warps);

  reg_reserved_type.resize(n_warps);
  reg_reserved_inst.resize(n_warps, vector<int>(200,0));
  num_cycles_decode_issue.resize(n_warps);
  reg_table_all_regs_used_list.resize(n_warps);
  arch_reg.resize(n_warps);
  physical_reg.resize(n_warps);
  write_regs_in_BB.resize(n_warps);
  m_gpu = gpu;
  //last_physical_reg = 3000;
  last_physical_reg.resize(n_warps,3000);
  free_regs.resize(n_warps);
  // set reg names across warps to push and pop from free list
  for(int i = 0; i<free_regs.size(); i++)
  {
    for (int j = 0;j<3000; j++)
    {
      int val = 3000 + j;
      free_regs[i].push(val);
    }
  }
}

void Scoreboard::resetValuesOfScoreboard(int warp_id)
{
  std::set<unsigned>::const_iterator it2;

  for (it2 = reg_table_all_regs_used_list[warp_id].begin(); it2 != reg_table_all_regs_used_list[warp_id].end(); it2++)
  {
    if (reg_table_all_regs_used[warp_id].find(*it2) != reg_table_all_regs_used[warp_id].end()) 
    {
      reg_table_all_regs_used[warp_id][*it2] = 0;
    }
  }
  
  warp_inst_issue_num[warp_id] = 0;
}


// Print scoreboard contents
void Scoreboard::printContents() const {
  printf("scoreboard contents (sid=%d): \n", m_sid);
  for (unsigned i = 0; i < reg_table.size(); i++) {
    if (reg_table[i].size() == 0) continue;
    printf("  wid = %2d: ", i);
    std::set<unsigned>::const_iterator it;
    for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
      printf("%u ", *it);
    printf("\n");
  }
}

void Scoreboard::reserveRegister(unsigned wid, unsigned regnum, bool gpgpu_perfect_mem_data, int pc, int m_cluster_id, int sid, int stalls_between_issues, int inst_num) {
  if (!(reg_table[wid].find(regnum) == reg_table[wid].end()) && !gpgpu_perfect_mem_data) {
    printf(
        "Error: trying to reserve an already reserved register (sid=%d, "
        "wid=%d, regnum=%d).",
        m_sid, wid, regnum);
    abort();
  }

  //reg_reserved[wid][regnum] = warp_inst_issue_num[wid];
  reg_reserved[wid][regnum] = inst_num;
  reg_pc[wid][regnum] = pc;
  SHADER_DPRINTF(SCOREBOARD, "Reserved Register - warp:%d, reg: %d\n", wid,
                 regnum);
  reg_table[wid].insert(regnum);
  reg_table_all_regs_used[wid][regnum] = 1;
  if (reg_table_all_regs_used_list[wid].find(regnum) == reg_table_all_regs_used_list[wid].end())
    reg_table_all_regs_used_list[wid].insert(regnum);
}


void Scoreboard::addWriteReg(unsigned wid, unsigned regnum) {
  write_regs_in_BB[wid].insert(regnum);
}

void Scoreboard::reserveRegisterWAR(unsigned wid, unsigned regnum, bool gpgpu_perfect_mem_data, int pc, int m_cluster_id, int sid, int stalls_between_issues, int inst_num) {
  // if (!(reg_table_WAR[wid].find(regnum) == reg_table_WAR[wid].end()) && !gpgpu_perfect_mem_data) {
  //   printf(
  //       "Error: trying to reserve an already reserved register (sid=%d, "
  //       "wid=%d, regnum=%d).",
  //       m_sid, wid, regnum);
  //   abort();
  // }
  // SHADER_DPRINTF(SCOREBOARD, "Reserved Register - warp:%d, reg: %d\n", wid,
  //                regnum);

  if(reg_reserved[wid].find(regnum) != reg_reserved[wid].end())
    reg_table_WAR[wid][regnum]+= 1;
  else
    reg_table_WAR[wid][regnum] = 1;
}

// Unmark register as write-pending
void Scoreboard::releaseRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) != reg_table[wid].end())) return;
  free_regs[wid].push(regnum);
  SHADER_DPRINTF(SCOREBOARD, "Release register - warp:%d, reg: %d\n", wid,
                 regnum);
  reg_table[wid].erase(regnum);
  longopregs[wid].erase(regnum);
}

void Scoreboard::releaseRegisterWAR(unsigned wid, unsigned regnum) {
  if (!(reg_table_WAR[wid].find(regnum) != reg_table_WAR[wid].end())) return;
  SHADER_DPRINTF(SCOREBOARD, "Release register - warp:%d, reg: %d\n", wid,
                 regnum);
  //reg_table_WAR[wid].erase(regnum);
  reg_table_WAR[wid][regnum]-= 1;
}

const bool Scoreboard::islongop(unsigned warp_id, unsigned regnum) {
  return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
}

const bool Scoreboard::islongop_hold(unsigned warp_id, const class warp_inst_t* inst) {

  std::set<int> inst_regs;

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::iterator it2;
  std::set<unsigned>::const_iterator it;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
  {
    inst_regs.insert(inst->out[iii]);
  }

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (longopregs[warp_id].find(*it2) != longopregs[warp_id].end()) {
      return true;
    }
  }
  return false;
}

void Scoreboard::reserveRegisters(const class warp_inst_t* inst, bool gpgpu_perfect_mem_data, int status, int m_cluster_id, int sid, int instNum, int numStalls) {
  warp_inst_issue_num[inst->warp_id()]++;
  bool dep_found = false;
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      dep_found = true;
      reserveRegister(inst->warp_id(), inst->out[r],gpgpu_perfect_mem_data,inst->pc, m_cluster_id, sid,numStalls,instNum);
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
    }
  }

  // reserve RAW registers 
  // for (unsigned r = 0; r < MAX_INPUT_VALUES; r++) {
  //   if (inst->in[r] > 0) {
  //     reserveRegisterWAR(inst->warp_id(), inst->in[r],gpgpu_perfect_mem_data,inst->pc, m_cluster_id, sid,numStalls,instNum);
  //     SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
  //                    inst->warp_id(), inst->in[r]);
  //     // if(inst->warp_id() == 0)
  //     //   std::cout<<"TRYING_TO_RESERVE_REG "<<inst->in[r]<<" VAL "<<reg_table_WAR[inst->warp_id()][inst->in[r]]<<" PC "<<inst->pc<<" OP "<<inst->op<<"\n";
  //   }
  // }


  // Keep track of long operations
  if (inst->is_load() && (inst->space.get_type() == global_space ||
                          inst->space.get_type() == local_space ||
                          inst->space.get_type() == param_space_kernel ||
                          inst->space.get_type() == param_space_local ||
                          inst->space.get_type() == param_space_unclassified ||
                          inst->space.get_type() == tex_space)) {
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      if (inst->out[r] > 0) {
        SHADER_DPRINTF(SCOREBOARD, "New longopreg marked - warp:%d, reg: %d\n",
                       inst->warp_id(), inst->out[r]);
        longopregs[inst->warp_id()].insert(inst->out[r]);
      }
    }
  }
}

void Scoreboard::addWriteRegs(const class warp_inst_t* inst, int wid)
{
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      if (inst->out[r] > 0) {
        addWriteReg(wid, inst->out[r]);
    }
  }
}

// Release registers for an instruction
void Scoreboard::releaseRegisters(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      SHADER_DPRINTF(SCOREBOARD, "Register Released - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
      releaseRegister(inst->warp_id(), inst->out[r]);
      longopregs[inst->warp_id()].erase(inst->out[r]);
    }
  }

  // for (unsigned r = 0; r < MAX_INPUT_VALUES; r++) {
  //   if (inst->in[r] > 0) {
  //     SHADER_DPRINTF(SCOREBOARD, "Register Released - warp:%d, reg: %d\n",
  //                    inst->warp_id(), inst->in[r]);
  //     releaseRegisterWAR(inst->warp_id(), inst->in[r]);
  //     longopregs[inst->warp_id()].erase(inst->in[r]);
  //     //  if(inst->warp_id() == 0)
  //     //   std::cout<<"%%TRYING_TO_REL_REG "<<inst->in[r]<<" VAL "<<reg_table_WAR[inst->warp_id()][inst->in[r]]<<" PC "<<inst->pc<<" OP "<<inst->op<<"\n";
  //   }
  // }

  
}

void Scoreboard::releaseAllBBRegs(int wid) {
  write_regs_in_BB[wid].clear();
}

void Scoreboard::RenameRegs(unsigned wid, class inst_t* inst, bool print, bool reg_renaming,std::vector<const warp_inst_t *> replayInst) {

  // check if WAW or WAR dependence and rename reg
  int max = 1000;
  int min = 100;
  
  total_inst_going_for_renaming++;
  bool reg_renamed = false;

  // collect all Write regs 
  std::set<int> inst_regs;
  for (unsigned iii = 0; iii < inst->outcount; iii++)
  {
    inst_regs.insert(inst->out[iii]);
  }

  std::set<int>::iterator it2;
  // if a Read Reg has a renamed dep- rename the read 
  std::set<int> inst_regs_in;
  for (unsigned iii = 0; iii < inst->incount; iii++)
  {
    inst_regs_in.insert(inst->in[iii]);
  }

  std::set<int> inst_replay_regs;
  // make list of all regs in replay list
  for(const class inst_t* ins : replayInst)
  {
    for (unsigned iii = 0; iii < ins->outcount; iii++)
      inst_replay_regs.insert(ins->out[iii]);

    for (unsigned jjj = 0; jjj < ins->incount; jjj++)
      inst_replay_regs.insert(ins->in[jjj]);
  }

  // check if a src register has been renamed, and rename it in the inst->in set as well
  int counter = 0;
  for (it2 = inst_regs_in.begin(); it2 != inst_regs_in.end(); it2++)
  {
    // direct the src reg to the correct renamed reg
    if(std::find(arch_reg[wid].begin(), arch_reg[wid].end(), inst->in[counter]) != arch_reg[wid].end()) 
    {
      auto it = find(arch_reg[wid].begin(), arch_reg[wid].end(), inst->in[counter]);
      int index = it - arch_reg[wid].begin();
      inst->in[counter] = physical_reg[wid][index];
    }
    // if there is no physical reg for this arch reg, mark one
    else
    {
      arch_reg[wid].push_back(inst->in[counter]);
      int physical_reg_num = free_regs[wid].front();
      free_regs[wid].pop();
      //physical_reg[wid].push_back(last_physical_reg[wid]);
      physical_reg[wid].push_back(physical_reg_num);
      inst->in[counter] = physical_reg_num;
    }
    counter++;
  }  
  
  counter = 0;
  // check for collision on destination registerrs
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    // check if any write reg has collision with rename list
    auto it = find(arch_reg[wid].begin(), arch_reg[wid].end(), (*it2));
    // rename the destination reg regardless of WAW or WAR issues, and do away woth all WAW and WAR hazards permanently
    // check if there is a collision with a architectural register file, then update the arch reg to phy reg mapping
    if (it != arch_reg[wid].end()) 
    {
      int index = it - arch_reg[wid].begin();
      if ((reg_table[wid].find(physical_reg[wid][index]) != reg_table[wid].end()) || (inst_replay_regs.find(physical_reg[wid][index]) != inst_replay_regs.end()))
      {
        int physical_reg_num = free_regs[wid].front();
        physical_reg[wid][index] = physical_reg_num;
        inst->out[counter] = physical_reg_num;
        free_regs[wid].pop();
      }
      else
      {
        inst->out[counter] = physical_reg[wid][index];
      }
    }
    // mapping does not exit, add it
    else
    {
      int physical_reg_num = free_regs[wid].front();
      free_regs[wid].pop();
      arch_reg[wid].push_back(inst->out[counter]);
      physical_reg[wid].push_back(physical_reg_num);
      inst->out[counter] = physical_reg_num;
    }
    counter++;
  }

  inst->register_renamed = 1;

  total_inst_renamed++;
}

/**
 * Checks to see if registers used by an instruction are reserved in the
 *scoreboard
 *
 * @return
 * true if WAW or RAW hazard (no WAR since in-order issue)
 **/
bool Scoreboard::checkCollision(unsigned wid, const class inst_t* inst, bool print, bool reg_renaming) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
  {
    inst_regs.insert(inst->out[iii]);
  }

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::iterator it2;
  std::set<unsigned>::const_iterator it;

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      return true;
    }
  }
  return false;
}

bool Scoreboard::checkIsIdempotent(unsigned wid, const class inst_t* inst) const {
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
  {
    inst_regs.insert(inst->out[iii]);
  }

  std::set<int>::iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (write_regs_in_BB[wid].find(*it2) != write_regs_in_BB[wid].end()) {
      return true;
    }
  }
  return false;

}

void Scoreboard::reg_values(unsigned wid) const {
  std::set<unsigned>::const_iterator it;
  for (it = reg_table[wid].begin(); it != reg_table[wid].end(); it++)
    std::cout <<(*it)<<" ";

  std::cout<<"LONG_PREGS ";
  for (it = longopregs[wid].begin(); it != longopregs[wid].end(); it++)
    std::cout <<(*it)<<" ";
}

bool Scoreboard::check_for_WAW_deps(unsigned wid, const class inst_t* inst) const {
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
  {
    inst_regs.insert(inst->out[iii]);
  }

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::iterator it2;
  std::set<unsigned>::const_iterator it;

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      return true;
    }
  }

  return false;
}

bool Scoreboard::check_WAR_or_WAW_replay(unsigned wid, const class inst_t* inst, std::vector<const warp_inst_t *> replayInst) const {
  std::set<int> inst_regs;
  std::set<int> inst_regs1;
  std::set<int> inst_replay_regs;
  bool WAR_WAW_found  = false;
  bool RAW_found = false;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  // for (unsigned jjj = 0; jjj < inst->incount; jjj++)
  //   inst_regs.insert(inst->in[jjj]);

  // if (inst->pred > 0) inst_regs.insert(inst->pred);
  // if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  // if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // make list of all regs in replay list
  for(const class inst_t* ins : replayInst)
  {
    for (unsigned iii = 0; iii < ins->outcount; iii++)
    {
      inst_replay_regs.insert(ins->out[iii]);
    }


    for (unsigned jjj = 0; jjj < ins->incount; jjj++)
      inst_replay_regs.insert(ins->in[jjj]);
  }

  // check for collision against replay queue instructions
  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (inst_replay_regs.find(*it2) != inst_replay_regs.end()) {
      WAR_WAW_found = true;
      break;
    }
  }
  
  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs1.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs1.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs1.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs1.insert(inst->ar2);
  for (it2 = inst_regs1.begin(); it2 != inst_regs1.end(); it2++)
  {
    if (inst_replay_regs.find(*it2) != inst_replay_regs.end()) {
      RAW_found = true;
      break;
    }
  }

  return (!RAW_found && WAR_WAW_found);

}

void Scoreboard::collisionInstPC(unsigned wid, const inst_t* inst, bool print, int pc, int m_cluster_id, int sid, std::vector<const warp_inst_t *> replayInst) {
  // Get list of all input and output registers
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
  {
    inst_regs.insert(inst->out[iii]);
  }

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::iterator it2;
  std::set<unsigned>::const_iterator it;

  std::set<int> inst_replay_regs;

  // make list of all regs in replay list
  for(const class inst_t* ins : replayInst)
  {
    for (unsigned iii = 0; iii < ins->outcount; iii++)
    {
      inst_replay_regs.insert(ins->out[iii]);
    }


    for (unsigned jjj = 0; jjj < ins->incount; jjj++)
      inst_replay_regs.insert(ins->in[jjj]);
  }

  std::cout <<"COLLISION STATUS "<<m_cluster_id<<" "<<sid<<" "<<wid<<" "<<pc<<" ";

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    int regnum = (*it2);
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      std::cout <<reg_pc[wid][regnum]<<" ";
    }
    else if (inst_replay_regs.find(*it2) != inst_replay_regs.end()) {
      std::cout <<reg_pc[wid][regnum]<<" ";
    }
  }
  std::cout <<"\n";
}

// bool Scoreboard::checkCollisionAddr(unsigned wid, const class inst_t* inst, bool print) const {
//   // Get list of all input and output registers
//   std::set<int> inst_regs;

//   for (unsigned iii = 0; iii < inst->addr_keeper_idx; iii++)
//   {
//     inst_regs.insert(inst->addr_keeper[iii]);
//   }

//   // Check for collision, get the intersection of reserved registers and
//   // instruction registers
//   std::set<int>::iterator it2;
//   std::set<unsigned>::const_iterator it;

//   for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
//   {
//     if (addr_table[wid].find(*it2) != addr_table[wid].end()) {
//       return true;
//     }
//   }
//   return false;
// }

void Scoreboard::get_dep_distance(unsigned wid, const warp_inst_t *inst1, int m_cluster_id, int sid, int stalls_between_issues, int OOO, int instNum, int numStalls)
{
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst1->outcount; iii++)
  {
    inst_regs.insert(inst1->out[iii]);
  }

  for (unsigned jjj = 0; jjj < inst1->incount; jjj++)
    inst_regs.insert(inst1->in[jjj]);

  if (inst1->pred > 0) inst_regs.insert(inst1->pred);
  if (inst1->ar1 > 0) inst_regs.insert(inst1->ar1);
  if (inst1->ar2 > 0) inst_regs.insert(inst1->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::iterator it2;
  std::set<unsigned>::const_iterator it;
}

void Scoreboard::get_closest_dependence(unsigned wid, const inst_t *inst1, bool print, int m_cluster_id, int sid, int OOO_dep,
int num_inst_OOO, unsigned stalls_between_issues, int isOOO) {
  std::set<int> inst_regs;
  int closest_dependence = -1;
  int decode_issue_distance = -1;
  int num_cycles_decode_issue_temp = -1;
  int num_cycles_decode_issue_ans = 0;
  inst_t* inst = const_cast<inst_t*>(inst1);

  int reg_cycle = -2;

  // only check if we are not starting the function again
  //if(warp_inst_issue_num[wid]!=0)
  {
    for (unsigned iii = 0; iii < inst->outcount; iii++)
    {
      int regnum = inst->out[iii];
      reg_cycle = reg_reserved[wid].find(regnum)->second;
      num_cycles_decode_issue_temp = num_cycles_decode_issue[wid].find(regnum)->second;

      if (reg_table_all_regs_used[wid].find(regnum) != reg_table_all_regs_used[wid].end()) 
      {
        if (reg_table_all_regs_used[wid].find(regnum)->second != 0)
        {
          if(reg_cycle > closest_dependence && reg_reserved[wid].find(regnum) != reg_reserved[wid].end())
          {
            closest_dependence = reg_cycle;
            num_cycles_decode_issue_ans = num_cycles_decode_issue_temp;
          }
        }
      }
    }

    for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    {
      int regnum = inst->in[jjj];
      reg_cycle = reg_reserved[wid].find(regnum)->second;
      num_cycles_decode_issue_temp = num_cycles_decode_issue[wid].find(regnum)->second;
      if (reg_table_all_regs_used[wid].find(regnum) != reg_table_all_regs_used[wid].end()) 
      {
        if (reg_table_all_regs_used[wid].find(regnum)->second != 0)
        {
          if(reg_cycle > closest_dependence && reg_reserved[wid].find(regnum) != reg_reserved[wid].end())
          {
            closest_dependence = reg_cycle;
            num_cycles_decode_issue_ans = num_cycles_decode_issue_temp;
          }
        }
      }
    }

    if (inst->pred > 0)
    {
      int regnum = inst->pred;
      reg_cycle = reg_reserved[wid].find(regnum)->second;
      num_cycles_decode_issue_temp = num_cycles_decode_issue[wid].find(regnum)->second;
      if (reg_table_all_regs_used[wid].find(regnum) != reg_table_all_regs_used[wid].end()) 
      {
        if (reg_table_all_regs_used[wid].find(regnum)->second != 0)
        {
          if(reg_cycle > closest_dependence && reg_reserved[wid].find(regnum) != reg_reserved[wid].end())
          {
            closest_dependence = reg_cycle;
            num_cycles_decode_issue_ans = num_cycles_decode_issue_temp;
          }
        }
      }
    }

    if (inst->ar1 > 0)
    {
      int regnum = inst->ar1;
      reg_cycle = reg_reserved[wid].find(regnum)->second;
      num_cycles_decode_issue_temp = num_cycles_decode_issue[wid].find(regnum)->second;
      if (reg_table_all_regs_used[wid].find(regnum) != reg_table_all_regs_used[wid].end()) 
      {
        if (reg_table_all_regs_used[wid].find(regnum)->second != 0)
        {
          if(reg_cycle > closest_dependence && reg_reserved[wid].find(regnum) != reg_reserved[wid].end())
          {
            closest_dependence = reg_cycle;
            num_cycles_decode_issue_ans = num_cycles_decode_issue_temp;
          }
        }
      }
    }

    if (inst->ar2 > 0)
    {
      int regnum = inst->ar2;
      reg_cycle = reg_reserved[wid].find(regnum)->second;
      num_cycles_decode_issue_temp = num_cycles_decode_issue[wid].find(regnum)->second;
      if (reg_table_all_regs_used[wid].find(regnum) != reg_table_all_regs_used[wid].end()) 
      {
        if (reg_table_all_regs_used[wid].find(regnum)->second != 0)
        {
          if(reg_cycle > closest_dependence && reg_reserved[wid].find(regnum) != reg_reserved[wid].end())
          {
            closest_dependence = reg_cycle;
            num_cycles_decode_issue_ans = num_cycles_decode_issue_temp;
          }
        }
      }
    }
  }

  int closest_num = 0;
  int num_cycles_decode_issue_final = num_cycles_decode_issue_ans;
  if(closest_dependence!= -1)
    closest_num = (warp_inst_issue_num[wid] + 1)  - closest_dependence;
  else 
    {
      closest_num = 0;
      num_cycles_decode_issue_final = 0;
    }

  // OOO_dep -> was inst dep on a DEB instruction?
  // stalls_between_issues -> Stalls between in order issues
  // num_inst_OOO -> OOO inst issued before this warp was issued
  std::cout <<"WARP_CLOSEST_DEPENDENCE "<<wid<<" "<<m_cluster_id<<" "<<sid<<" "<<closest_num <<" "
  <<num_cycles_decode_issue_final<<" "<<OOO_dep<<" "<<stalls_between_issues<<" "<<num_inst_OOO<<" "<<isOOO<<"\n";
}

void Scoreboard::set_num_cycles_deocode_issue(int num_cycles, const class warp_inst_t* inst)
{
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      num_cycles_decode_issue[inst->warp_id()][inst->out[r]] = num_cycles;
    }
  }
}

bool Scoreboard::pendingWrites(unsigned wid) const {
  return !reg_table[wid].empty();
}

bool Scoreboard::pendingWrites(unsigned wid, bool ignore) const {
  if (ignore) return false;
  return !reg_table[wid].empty();
}

/* Added Functions */
/* Check if instructions collide with an instruction in the replay queue
*  @return
* true if WAR or WAW hazard
*/
bool Scoreboard::checkReplayCollision(unsigned wid, const class inst_t* inst, std::vector<const warp_inst_t *> replayInst) const {
  std::set<int> inst_regs;
  std::set<int> inst_replay_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // make list of all regs in replay list
  for(const class inst_t* ins : replayInst)
  {
    for (unsigned iii = 0; iii < ins->outcount; iii++)
    {
      inst_replay_regs.insert(ins->out[iii]);
    }


    for (unsigned jjj = 0; jjj < ins->incount; jjj++)
      inst_replay_regs.insert(ins->in[jjj]);
  }

  // check for collision against replay queue instructions
  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (inst_replay_regs.find(*it2) != inst_replay_regs.end()) {
      return true;
    }
  }
  return false;
}


int Scoreboard::getIndependentInstructionCount(unsigned wid, const class inst_t* inst, std::vector<const warp_inst_t *> replayInst) const {
  std::set<int> inst_regs;
  std::set<int> inst_replay_regs;

  int independent_inst = 0;
  int dependence_found = 0;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // make list of all regs in replay list
  for(const class inst_t* ins : replayInst)
  {
    dependence_found = 0;
    inst_replay_regs.clear();
    for (unsigned iii = 0; iii < ins->outcount; iii++)
    {
      inst_replay_regs.insert(ins->out[iii]);
    }

    for (unsigned jjj = 0; jjj < ins->incount; jjj++)
      inst_replay_regs.insert(ins->in[jjj]);


    // check for collision against replay queue instructions
    std::set<int>::const_iterator it2;
    for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
    {
      if (inst_replay_regs.find(*it2) != inst_replay_regs.end()) {
        dependence_found = 1;
        break;
      }
    }
    if(dependence_found == 0)
    {
      independent_inst++;
    }
  }
  return independent_inst;
}

//unmark registers as write Pending for mem operations
void Scoreboard::releaseRegisterMem(unsigned wid, unsigned regnum,int val,int op, int type) {
  if (!(reg_table_mem[wid].find(regnum) != reg_table_mem[wid].end())) return;
  reg_table_mem[wid].erase(regnum);
  reg_released_mem[wid][regnum] = cycles_passed;
}

//unmark registers as write pending for comp operations
void Scoreboard::releaseRegisterComp(unsigned wid, unsigned regnum) {
  if (!(reg_table_comp[wid].find(regnum) != reg_table_comp[wid].end())) return;
  reg_table_comp[wid].erase(regnum);

  reg_released_comp[wid][regnum] = cycles_passed;
  //comp_reg_reserve_cycle = comp_reg_reserve_cycle + (cycles_passed - reg_reserved_comp[wid][regnum]);
}

void Scoreboard::reserveRegisterMem(unsigned wid, unsigned regnum, bool is_load) {
  reg_table_mem[wid].insert(regnum);
  reg_reserved_mem[wid][regnum] = cycles_passed;
  reg_reserved_type[wid][regnum] = 1;
}

void Scoreboard::reserveRegisterComp(unsigned wid, unsigned regnum) {
  reg_table_comp[wid].insert(regnum);
  reg_reserved_comp[wid][regnum] = cycles_passed;
  reg_reserved_type[wid][regnum] = 2;
}

void Scoreboard::reserveRegistersMem(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      reserveRegisterMem(inst->warp_id(), inst->out[r], inst->is_load());
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
    }
  }
}

void Scoreboard::reserveRegistersComp(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      reserveRegisterComp(inst->warp_id(), inst->out[r]);
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
    }
  }
}

// Release registers for a mem instruction
void Scoreboard::releaseRegistersMem(const class warp_inst_t* inst,int val) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      releaseRegisterMem(inst->warp_id(), inst->out[r],val,inst->op,inst->space.get_type());
    }
  }
}

// Release registers for a comp instruction
void Scoreboard::releaseRegistersComp(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      releaseRegisterComp(inst->warp_id(), inst->out[r]);
    }
  }
}

void Scoreboard::appendMemStatus(warp_inst_t &inst,int type)
{
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst.out[r] > 0) {
      reg_type_mem[inst.warp_id()][inst.out[r]] = type;
      reg_load_type[inst.warp_id()][inst.out[r]] = inst.is_load();
    }
  }
}

bool Scoreboard::pendingWritesMem(unsigned wid) const {
  return !reg_table_mem[wid].empty();
}

bool Scoreboard::pendingWritesComp(unsigned wid) const {
  return !reg_table_comp[wid].empty();
}

std::vector<int> Scoreboard::checkCollisionMem(unsigned wid, const class inst_t* inst) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;
  std::vector<int> result;
  result.resize(3);

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;

  // check for longest taking release Reg and print the corresponding reserve and release cycle with the register number
  int reserve_c = -1;
  int release_c = -1;
  result[1] = reserve_c;
  result[2] = release_c;

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (reg_table_mem[wid].find(*it2) != reg_table_mem[wid].end()) {
	    result[0] = 1;
      return result;
    }
  }

  result[0] = 0;
  return result;
}

std::vector<int> Scoreboard::checkCollisionComp(unsigned wid, const class inst_t* inst) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;
  std::vector<int> result;
  result.resize(3);

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;

  // check for longest taking release Reg and print the corresponding reserve and release cycle with the register number
  int reserve_c = -1;
  int release_c = -1;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    unsigned regnum = *it2;
    if (reg_reserved_comp[wid].count(regnum) == 0) continue;
    int reserve = reg_reserved_comp[wid].find(regnum)->second;

    if (reg_released_comp[wid].count(regnum) == 0) continue;
    int release = reg_released_comp[wid].find(regnum)->second;

    // Take latest reservation that was released
    if (reserve <= release && release > release_c)
    {
      reserve_c = reserve;
      release_c = release;
    }
  }
  result[1] = reserve_c;
  result[2] = release_c;

    bool mem_data_col = false;
  bool comp_data_col = false;

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    unsigned regnum = *it2;
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      if (reg_reserved_type[wid].find(regnum)->second == 1)
        mem_data_col = true;
      if (reg_reserved_type[wid].find(regnum)->second == 2)
        comp_data_col = true;
    }
  }

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (reg_table_comp[wid].find(*it2) != reg_table_comp[wid].end()) {
      result[0] = 1;
      return result;
    }
  }
  result[0] = 0;
  return result;
}

bool Scoreboard::checkConsecutiveInstIndep(const class inst_t* inst, const class inst_t *inst1) const{

  std::set<int> inst_regs;
  std::set<int> inst_regs1;
  int a = 0;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  for (unsigned iii = 0; iii < inst1->outcount; iii++)
    inst_regs1.insert(inst1->out[iii]);

  for (unsigned jjj = 0; jjj < inst1->incount; jjj++)
    inst_regs1.insert(inst1->in[jjj]);

  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (inst_regs1.find(*it2) != inst_regs1.end()) {
      return false;
    }
  }
  return true;
}


