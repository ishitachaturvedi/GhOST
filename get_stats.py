# %%
from enum import Enum
import glob, os
from os import listdir
from os.path import isfile, join
from os import walk
import statistics
import matplotlib.pyplot as plt
import numpy as np
import statistics
from statistics import mean

class Instruction:
    def __init__(self, name, depends_on=None):
        self.name = name
        self.depends_on = depends_on or set()

    def __repr__(self):
        return self.name
    
# rename regs and return renamed BB
def renameRegisters (Renaming_on,inst_list,regs_used_set):
    
    # all regs read/written to in this BB 
    total_read_regs = []
    total_write_regs = []
    
    total_read_regs_arch = []
    total_write_regs_arch = []
    
    # per inst read/write
    read_regs_per_inst = []
    write_regs_per_inst = []
    
    # keeps track of waw/war/raw register dependencies
    raw_reg_deps_per_inst = []
    war_reg_deps_per_inst = []
    waw_reg_deps_per_inst = []
    
    # keeps track of inst #s which have a waw/war/raw dependence with current instruction
    raw_instnum_deps_per_inst = []
    waw_instnum_deps_per_inst = []
    war_instnum_deps_per_inst = []
    
    # instructions with no dependence
    independent_inst_found = 0
    independent_inst_found_waw_war = 0
    total_inst_found = 0
    
    # keeps track of inst numbers to know which inst the current inst depends on
    counter = 0
    instnum = []
    
    # instructions with no dependence
    
    false_deps_list = []
    
    inst_with_only_false_deps = 0
    
    free_list = []
    renamed_register = []
    renamed_to = []
    renamed_inst_list = []
    
    # generate free list
    for i in range (0,255):
        reg = "R"+str(i)
        free_list.append(reg)
    
    for inst in inst_list:
        read_regs = []
        write_regs = []
        write_regs_arch = []
        read_regs_arch = []
        
        raw_deps = []
        war_deps = []
        waw_deps = []
        raw_deps_arch = []
        war_deps_arch = []
        waw_deps_arch = []
        all_regs_phys = []
        all_regs_arch = []
        renamed_regs = []
        
        inst = inst.split(' ')
        num_write = int(inst[2])
        num_read = int(inst[4+num_write])
        
        waw_dependence = False
        war_dependence = False
        raw_dependence = False 
        
        waw_dependence_arch = False
        war_dependence_arch = False
        raw_dependence_arch = False 
        
        instnum.append(counter)
        counter = counter + 1
        
        # inst nums with WAW dependence
        raw_instnum = []
        waw_instnum = []
        war_instnum = []
        
        for i in range(num_write):
            # check for WAW dependence
            reg_id_orig = inst[3+i]
            
            # rename this to get the correct reg id
            if reg_id_orig in renamed_register and reg_id_orig !="R255":
                idx =  renamed_register.index(reg_id_orig)
                reg_id = renamed_to[idx]
            elif reg_id_orig!="R255":
                renamed_register.append(reg_id_orig)
                reg_id = free_list.pop(0)
                renamed_to.append(reg_id)
                regs_used_set.add(reg_id)
                free_list.append(reg_id)
            else:
                reg_id = "R255"
                
            all_regs_arch.append(reg_id_orig)
                
            write_regs.append(reg_id)
            
            write_regs_arch.append(reg_id_orig)
                    
            all_regs_phys.append(reg_id)
            
            if(reg_id in total_write_regs):
                waw_dependence = True
                waw_deps.append(reg_id)
            
            if(reg_id_orig in total_write_regs_arch):
                waw_dependence_arch = True
                waw_deps_arch.append(reg_id_orig)
            
            # go over write_regs_per_inst to find if we have a waw dependence and store the inst with the waw dependence
            for inst_iter in range(len(write_regs_per_inst)):
                if reg_id in write_regs_per_inst[inst_iter]:
                    waw_instnum.append(inst_iter)
                
            # check for WAR dependence
            if(reg_id in total_read_regs):
                war_dependence = True
                war_deps.append(reg_id)
                
            if(reg_id_orig in total_read_regs_arch):
                war_dependence_arch = True
                war_deps_arch.append(reg_id_orig)
                
            # go over read_regs_per_inst to find if we have a war dependence and store the inst with the waw dependence
            for inst_iter in range(len(read_regs_per_inst)):
                if reg_id in read_regs_per_inst[inst_iter]:
                    war_instnum.append(inst_iter)
        
        read_idx = 5 + num_write
        for i in range(num_read):
            # check for RAW hazard
            
            reg_id_orig = inst[read_idx+i]
            # rename this to get the correct reg id
            if reg_id_orig in renamed_register and reg_id_orig !="R255":
                idx =  renamed_register.index(reg_id_orig)
                reg_id = renamed_to[idx]
            elif reg_id_orig!="R255":
                renamed_register.append(reg_id_orig)
                reg_id = free_list.pop(0)
                renamed_to.append(reg_id)
                regs_used_set.add(reg_id)
                free_list.append(reg_id)
            else:
                reg_id = "R255"
                
            read_regs.append(reg_id)
            read_regs_arch.append(reg_id_orig)
            all_regs_phys.append(reg_id)
            all_regs_arch.append(reg_id_orig)
                    
            if(reg_id in total_write_regs):
                raw_dependence = True
                raw_deps.append(reg_id)
                
            if(reg_id_orig in total_write_regs_arch):
                raw_dependence = True
                raw_deps_arch.append(reg_id_orig)
                
            # go over read_regs_per_inst to find if we have a raw dependence and store the inst with the waw dependence
            for inst_iter in range(len(read_regs_per_inst)):
                if reg_id in read_regs_per_inst[inst_iter]:
                    raw_instnum.append(inst_iter)
            
        # do this separately, so an inst does not have a dependece with itself in our check
        for temp_reg in write_regs:
            total_write_regs.append(temp_reg)
        for temp_reg in read_regs:
            total_read_regs.append(temp_reg)
            
        for temp_reg in write_regs_arch:
            total_write_regs_arch.append(temp_reg)
        for temp_reg in read_regs_arch:
            total_read_regs_arch.append(temp_reg)
            
        read_regs_per_inst.append(read_regs)
        write_regs_per_inst.append(write_regs)
        
        # for regs which have waw/war dependencies 
        # eg:
        # 1. R5 = R4 + R5
        # 2. R3 = R1 + R2
        # 3. R2 = R4 + R5
        # Here 3 should be marked as a WAR dependence with 2. It also has a RAW dependence with 1, but that does not remove the WAR dependence to 1
        waw_war_deps_found = False
        temp_reg_renamed = []
        
        #print("LOOKING at inst ",inst," waw_dependence ",waw_dependence," war_dependence ",war_dependence,end="")
        for reg in waw_deps:
            #print("reg in waw ",reg,end="")
            if reg not in raw_deps and reg not in temp_reg_renamed:
                #print("WAW_DEPS_FOUND ",reg,end="")
                waw_war_deps_found = True
                false_deps_list.append(inst)
                temp_reg_renamed.append(reg)                
        for reg in war_deps:
            #print("reg in war ",reg,end="")
            if reg not in raw_deps and reg not in temp_reg_renamed:
                #print("WAR_DEPS_FOUND ",reg,end="")
                waw_war_deps_found = True
                temp_reg_renamed.append(reg)
                false_deps_list.append(inst)
        #print()
                
        # so we dont double rename a register
        temp_reg_renamed = []
        # rename away the false dependencies
        #print("**************************")
        #print("looking at insn ",inst[0]," regs ",len(all_regs_phys)," inst ",inst)
        for reg_idx in range(len(all_regs_phys)):
            # if this register does not have a RAW hazard:
            reg_phys = all_regs_phys[reg_idx]
            reg_arch = all_regs_arch[reg_idx]
            #print("reg_arch ",reg_arch," reg_phys ",reg_phys," no raw ",(reg_arch not in raw_deps_arch)," war ",(reg_arch in war_deps_arch)," waw ",(reg_arch in waw_deps_arch)," temprename ",(reg_phys not in temp_reg_renamed))
            #if reg_phys not in raw_deps and (reg_phys in war_deps or reg_phys in waw_deps) and reg_phys not in temp_reg_renamed and reg_phys!="R255":
            if reg_arch not in raw_deps_arch and (reg_arch in war_deps_arch or reg_arch in waw_deps_arch) and reg_arch not in temp_reg_renamed and reg_arch!="R255":
                # this register has false dependencies, rename it
                idx = renamed_register.index(reg_arch)
                assert(len(free_list) > 1)
                new_reg =  free_list.pop(0)
                start_reg = new_reg
                regrenamed = False
                while (regrenamed == False):
                    if new_reg not in total_write_regs and new_reg not in total_read_regs:
                        regs_used_set.add(new_reg)
                        renamed_to[idx] = new_reg
                        temp_reg_renamed.append(reg_phys)
                        free_list.append(new_reg) 
                        regrenamed =  True
                    else:
                        free_list.append(new_reg) 
                        new_reg = free_list.pop(0)
                        if new_reg == start_reg:
                            print("ERROR ISHITA !! ",new_reg)
                            exit(0)
                #print("RENAMING orig ",reg_arch," new ",new_reg," line ",inst)
                
        # get new inst after renaming:
        line = str(inst[0]) + str(" ") + str(inst[1]) + str(" ") + str(inst[2])+ str(" ")
        for i in range(num_write):
            reg = (inst[3+i])
            if(reg!="R255"):
                idx = renamed_register.index(reg)
                renamed = renamed_to[idx]
            else:
                renamed = "R255"
            line =line + str(renamed) + str(" ")
        next_read = 3 + num_write 
        line = line + str(inst[next_read]) + str (" ") + str(inst[next_read+1]) + str(" ")
        read_idx = 5 + num_write
        for i in range(num_read):
            reg = (inst[read_idx+i])
            if(reg!="R255"):
                idx = renamed_register.index(reg)
                renamed = renamed_to[idx]
            else:
                renamed = "R255"
            line =line + str(renamed) + " "
        final_read = read_idx + num_read
        for i in range(final_read,len(inst)):
            line = line + str(inst[i]) + str(" ")
        #print("******* Renamed ",line)
        line = line + "\n"
        renamed_inst_list.append(line)
        
        independent_inst_found_waw_war = independent_inst_found_waw_war + (1 & (not waw_war_deps_found))
        inst_with_only_false_deps = inst_with_only_false_deps + (1 & waw_war_deps_found)
            
        raw_reg_deps_per_inst.append(raw_deps)
        waw_reg_deps_per_inst.append(waw_deps)
        war_reg_deps_per_inst.append(war_deps)
        
        raw_instnum_deps_per_inst.append(raw_instnum)
        waw_instnum_deps_per_inst.append(waw_instnum)
        war_instnum_deps_per_inst.append(war_instnum)
        
    # return the dependence graph and the number of independent instructions
    return independent_inst_found, total_inst_found, raw_reg_deps_per_inst, waw_reg_deps_per_inst, war_reg_deps_per_inst,raw_instnum_deps_per_inst,waw_instnum_deps_per_inst,war_instnum_deps_per_inst, independent_inst_found_waw_war, inst_with_only_false_deps, false_deps_list, renamed_inst_list, regs_used_set

# generate a dependence graph for all
def get_dependence_graph(Renaming_on,inst_list):
    
    # all regs read/written to in this BB 
    total_read_regs = []
    total_write_regs = []
    
    # per inst read/write
    read_regs_per_inst = []
    write_regs_per_inst = []
    
    # keeps track of waw/war/raw register dependencies
    raw_reg_deps_per_inst = []
    war_reg_deps_per_inst = []
    waw_reg_deps_per_inst = []
    
    # keeps track of inst #s which have a waw/war/raw dependence with current instruction
    raw_instnum_deps_per_inst = []
    waw_instnum_deps_per_inst = []
    war_instnum_deps_per_inst = []
    
    # keeps track of inst numbers to know which inst the current inst depends on
    instnum = []
    
    # instructions with no dependence
    independent_inst_found = 0
    independent_inst_found_waw_war = 0
    total_inst_found = 0
    
    counter = 0
    
    false_deps_list = []
    
    inst_with_only_false_deps = 0
    
    for inst in inst_list:
        read_regs = []
        write_regs = []
        
        raw_deps = []
        war_deps = []
        waw_deps = []
        
        inst = inst.split(' ')
        num_write = int(inst[2])
        num_read = int(inst[4+num_write])
        
        waw_dependence = False
        war_dependence = False
        raw_dependence = False 
        
        instnum.append(counter)
        counter = counter + 1
        
        # inst nums with WAW dependence
        raw_instnum = []
        waw_instnum = []
        war_instnum = []
        
        for i in range(num_write):
            write_regs.append(inst[3+i])
            # check for WAW dependence
            reg_id = inst[3+i]
            if(reg_id in total_write_regs):
                waw_dependence = True
                waw_deps.append(reg_id)
            
            # go over write_regs_per_inst to find if we have a waw dependence and store the inst with the waw dependence
            for inst_iter in range(len(write_regs_per_inst)):
                if reg_id in write_regs_per_inst[inst_iter]:
                    waw_instnum.append(inst_iter)
                
            # check for WAR dependence
            if(reg_id in total_read_regs):
                war_dependence = True
                war_deps.append(reg_id)
                
            # go over read_regs_per_inst to find if we have a war dependence and store the inst with the waw dependence
            for inst_iter in range(len(read_regs_per_inst)):
                if reg_id in read_regs_per_inst[inst_iter]:
                    war_instnum.append(inst_iter)
        
        read_idx = 5 + num_write
        for i in range(num_read):
            read_regs.append(inst[read_idx+i])
            # check for RAW hazard
            reg_id = inst[read_idx+i]
            if(reg_id in total_write_regs):
                raw_dependence = True
                raw_deps.append(reg_id)
                
            # go over read_regs_per_inst to find if we have a raw dependence and store the inst with the waw dependence
            for inst_iter in range(len(read_regs_per_inst)):
                if reg_id in read_regs_per_inst[inst_iter]:
                    raw_instnum.append(inst_iter)
            
        # do this separately, so an inst does not have a dependece with itself in our check
        for i in range(num_write):
            total_write_regs.append(inst[3+i])
        for i in range(num_read):
            total_read_regs.append(inst[read_idx+i])
            
        read_regs_per_inst.append(read_regs)
        write_regs_per_inst.append(write_regs)
        
        dependence_found = (waw_dependence & (not Renaming_on)) | (war_dependence & (not Renaming_on)) | raw_dependence
        independent_inst_found = independent_inst_found + (1 & (not dependence_found))
        total_inst_found = total_inst_found + 1
        
        
        # for regs which have waw/war dependencies 
        # eg:
        # 1. R5 = R4 + R5
        # 2. R3 = R1 + R2
        # 3. R2 = R4 + R5
        # Here 3 should be marked as a WAR dependence with 2. It also has a RAW dependence with 1, but that does not remove the WAR dependence to 1
        waw_war_deps_found = False
        for reg in waw_deps:
            if reg not in raw_deps:
                waw_war_deps_found = True
                false_deps_list.append(inst)
        if not waw_war_deps_found:
            for reg in war_deps:
                if reg not in raw_deps:
                    waw_war_deps_found = True
                    
        independent_inst_found_waw_war = independent_inst_found_waw_war + (1 & (not waw_war_deps_found))
        inst_with_only_false_deps = inst_with_only_false_deps + (1 & waw_war_deps_found)
        
        raw_reg_deps_per_inst.append(raw_deps)
        waw_reg_deps_per_inst.append(waw_deps)
        war_reg_deps_per_inst.append(war_deps)
        
        raw_instnum_deps_per_inst.append(raw_instnum)
        waw_instnum_deps_per_inst.append(waw_instnum)
        war_instnum_deps_per_inst.append(war_instnum)
        
        # if(waw_war_deps_found):
        #     print(inst)
        
    # return the dependence graph and the number of independent instructions
    return independent_inst_found, total_inst_found, raw_reg_deps_per_inst, waw_reg_deps_per_inst, war_reg_deps_per_inst,raw_instnum_deps_per_inst,waw_instnum_deps_per_inst,war_instnum_deps_per_inst, independent_inst_found_waw_war, inst_with_only_false_deps, false_deps_list

def generate_all_permutations(instructions, current_permutation, remaining_instructions, valid_permutations):
    if not remaining_instructions:
        valid_permutations.append(current_permutation[:])  # Add a copy of the current permutation
        return
    
    if(len(valid_permutations)==1000):
        return

    for instr in remaining_instructions[:]:  # Use a copy of the list for iteration
        # Check if instr can be added to the current permutation without violating dependencies
        if all(dep in current_permutation for dep in instr.depends_on):
            current_permutation.append(instr)
            remaining_instructions.remove(instr)

            # Recursively generate permutations
            generate_all_permutations(instructions, current_permutation, remaining_instructions, valid_permutations)

            # Backtrack: Restore the state of remaining_instructions
            current_permutation.pop()
            remaining_instructions.append(instr)

def get_possible_reorderings(inst_list,raw_instnum_deps_per_inst,waw_instnum_deps_per_inst,war_instnum_deps_per_inst,Renaming_on):
    
    # generate the Instruction class based on renaming status
    instructions = []
    
    for i in range(len(inst_list)):
        dependence = set()
        # add all raw dependencies
        for instnum in raw_instnum_deps_per_inst[i]:
            dependence.add(instructions[instnum])
        # add war and waw dependencies if renaming is off
        if not Renaming_on:
            for instnum in waw_instnum_deps_per_inst[i]:
                dependence.add(instructions[instnum])
            for instnum in war_instnum_deps_per_inst[i]:
                dependence.add(instructions[instnum])
                
        # add this inst to the dependence_list
        instructions.append(Instruction(str(i),depends_on=dependence))
        
    # start generating all possible permutations
    valid_permutations = []
    generate_all_permutations(instructions, [], instructions.copy(), valid_permutations)
    
    return len(valid_permutations)

def find_first_difference(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        # Read lines from both files
        lines1 = file1.readlines()
        lines2 = file2.readlines()

        # Find the minimum number of lines between the two files
        min_lines = min(len(lines1), len(lines2))

        # Iterate through the lines and find the first difference
        for i in range(min_lines):
            lines1[i] = lines1[i].strip()
            lines1[i] =  lines1[i].split(' ')
            lines2[i] = lines2[i].strip()
            lines2[i] =  lines2[i].split(' ')
            if lines1[i] != lines2[i]:
                print(f"First difference found at line {i + 1}:\nFile 1: {lines1[i]}File 2: {lines2[i]}")
                return

        # If all lines match up to the length of the shorter file, check for extra lines
        if len(lines1) != len(lines2):
            print(f"Files have different lengths. First difference found at line {min_lines + 1} or later.")
        else:
            print("Files are identical.")

def remove_false_dependencies():
    
    # Example usage
    # find_first_difference("/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/ISCA-24-Bmrks/lavaMD_rodinia/kernel-1.traceg", "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/ISCA-24-Bmrks/lavaMD_rodinia/kernel-1_rename.traceg")
    
    # exit(0)
    
    control_inst = [
        #"BRA",
        "BRX",
        "JMP",
        "JMX",
        "SSY",
        #"BAR.SYNC",
        "CAL",
        "JCAL",
        "PRET",
        "RET",
        "BRK",
        "PBK",
        "CONT",
        "PCNT",
        "EXIT",
        "PEXIT",
        "BPT"       
    ]
    
    branch = ["BRA"]
    
    bmrks =[
        #"lavaMD",
        #"lavaMD_unroll",
        #"convolution_unroll"
        "convolution"
        ]
    
    traces = [
        #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/ISCA-24-Bmrks/lavaMD_rodinia/kernel-1.traceg"],
        #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/lavaMD_unroll/kernel-1.traceg"],
        #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/convolution_unroll/kernel-1.traceg",
        #"/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/convolution_unroll/kernel-2.traceg"]
        ["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-1.traceg",
        "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-2.traceg",
        "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-51.traceg",
        "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-52.traceg",]
    ] 
    
    write_traces = [
        #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/ISCA-24-Bmrks/lavaMD_rodinia/kernel-1_rename.traceg"],
        #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/lavaMD_unroll/kernel-1_rename.traceg"],
        #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/convolution_unroll/kernel-1_rename.traceg",
        #"/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/convolution_unroll/kernel-2_rename.traceg"],
        ["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-1_rename.traceg",
        "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-2_rename.traceg",
        "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-51_rename.traceg",
        "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-52_rename.traceg",]
    ] 
    
    branch_counter = 0
    
    for k in range(len(traces)):
        print("LOOKING AT HERE ",bmrks[k])
        inst_list = []
        size_of_BB = []
        independent_instructions = []
        independent_instructions_waw_war = []
        inst_with_WAW_WAR = []
        number_of_reorderings = []
        BB_count = 0
        for file_idx in range(len(traces[k])):
            file = traces[k][file_idx]
            fin = open(file,"r")
            write_file = write_traces[k][file_idx]
            fout = open(write_file,"w")
            read_file =  False
            regs_used_set = set()
            
            line_count = 0
            
            printing = 0
            
            # read file till a control instruction is found
            for line in fin:
                line_orig = line
                line = line.strip()
                line1 =  line.split(' ')
                
                inst_type = 0
                # if control instruction is found, then calculate # dependence free instructions and the # of possible reorderings
                if read_file and (len(line1) > 4):
                    num_write =  int(line1[2])
                    inst_type = line1[3 + num_write]
                
                if inst_type in branch:
                    branch_counter = branch_counter + 1
                
                if read_file and (inst_type in control_inst or (inst_type in branch and branch_counter == 20)):
                    branch_counter = 0
                    inst_list.append(line)
                    #print("WRITING TO FILE CONTROL INST ",inst_type)
                    
                    independent_inst_found,total_inst_found,raw_reg_deps_per_inst,waw_reg_deps_per_inst,war_reg_deps_per_inst,raw_instnum_deps_per_inst,waw_instnum_deps_per_inst,war_instnum_deps_per_inst,independent_inst_found_waw_war,inst_with_only_false_deps, false_deps_list, renamed_list, regs_used_set = renameRegisters(False,inst_list,regs_used_set)
                    
                    num_reorderings = 0
                    BB_count = BB_count + 1
                        
                    # if(inst_with_only_false_deps):
                    #     print("**********************************")
                    #     print("#################FALSE DEPS INST")
                    #     for inst in false_deps_list:
                    #         print(inst)
                    #     print("#################BASIC BLOCK")
                    #     for inst in inst_list:
                    #         print(inst)
                    #     print("@@@@@@@@@@@@ RENAMED BASIC BLOCK")
                    #     for inst in renamed_list:
                    #         print(inst)
                    
                    if(len(renamed_list) != len(inst_list)):
                        print("HUGE ERROR ",len(renamed_list)," ", len(inst_list))
                    
                    for writeline in renamed_list:
                        linetowrite = writeline
                        fout.write(linetowrite)
                        
                    inst_list = []
                elif read_file:
                    inst_list.append(line)
                else:
                    inst_list.append(line)
                
                if("warp" in  line):
                    read_file = False
                    
                if("inst" in line):
                    read_file = True
                    for writeline in inst_list:
                        linetowrite = writeline + "\n"
                        fout.write(linetowrite)
                    inst_list = []
                    
                line_count = line_count + 1
                
            for writeline in inst_list:
                linetowrite = writeline + "\n"
                fout.write(linetowrite)
        
            inst_list = []
                    
            print("TOTAL REGS USED ",len(regs_used_set))
    
def get_WAW_WAR_stats(Renaming_on):
    
    control_inst = [
        "BRA",
        "BRX",
        "JMP",
        "JMX",
        "SSY",
        "BAR.SYNC",
        "CAL",
        "JCAL",
        "PRET",
        "RET",
        "BRK",
        "PBK",
        "CONT",
        "PCNT",
        "EXIT",
        "PEXIT",
        "BPT"       
    ]
    
    
    # bmrks =[
    #     # "mandelbort", # 38%
    #     # "particlefinder_float", #23% branch stuff
    #     # "kmeans-altis", # 20% memory low-occupancy
    #     # "RAY", #10% high-occupancy
    #     # "lud", # 5% speedup low occupancy
    #     # "fw", # 4% speedup high occupancy
    #     # "pathfinder" #1%
    #     #"lavaMD",
    #     "lavaMD_unroll",
    #     #"convolution",
    #     #"convolution_unroll"
    #     ]
    
    # traces = [
    #     #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/ISCA-24-Bmrks/lavaMD_rodinia/kernel-1.traceg"],
    #     ["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/lavaMD_unroll/kernel-1.traceg"],
    #     #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-1.traceg",
    #     #"/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/pagoda_traces/traces_convolution/kernel-29.traceg"],
    #     #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/convolution_unroll/kernel-1.traceg",
    #     # "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/convolution_unroll/kernel-2.traceg"]
    # ] 
    
    bmrks =[
        #"lavaMD",
        #"lavaMD_unroll",
        "convolution_unroll"
        ]
    
    traces = [
        #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/ISCA-24-Bmrks/lavaMD_rodinia/kernel-1_rename.traceg"],
        #["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/lavaMD_unroll/kernel-1_rename.traceg"],
        ["/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/convolution_unroll/kernel-1_rename.traceg",
        "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/unrolled_kernels/convolution_unroll/kernel-2_rename.traceg"]
    ] 
    
    print("VALS ",len(bmrks)," ",len(traces))
    for k in range(len(traces)):
        print("LOOKING AT HERE ",bmrks[k])
        inst_list = []
        size_of_BB = []
        independent_instructions = []
        independent_instructions_waw_war = []
        inst_with_WAW_WAR = []
        number_of_reorderings = []
        BB_count = 0
        for file in traces[k]:
            fin = open(file,"r")
            read_file =  False
            
            line_count = 0
            
            printing = 0
            
            # read file till a control instruction is found
            for line in fin:
                line = line.strip()
                line1 =  line.split(' ')
                
                inst_type = 0
                # if control instruction is found, then calculate # dependence free instructions and the # of possible reorderings
                if read_file and (len(line1) > 4):
                    num_write =  int(line1[2])
                    inst_type = line1[3 + num_write]
                
                # if(line_count%50000 == 0):
                #     print("line_count ",line_count)
                
                if read_file and inst_type in control_inst:
                    independent_inst_found,total_inst_found,raw_reg_deps_per_inst,waw_reg_deps_per_inst,war_reg_deps_per_inst,raw_instnum_deps_per_inst,waw_instnum_deps_per_inst,war_instnum_deps_per_inst,independent_inst_found_waw_war,inst_with_only_false_deps, false_deps_list = get_dependence_graph(Renaming_on,inst_list)
                    
                    num_reorderings = 0
                    # num_reorderings = get_possible_reorderings(inst_list,raw_instnum_deps_per_inst, waw_instnum_deps_per_inst, war_instnum_deps_per_inst,Renaming_on)
                    BB_count = BB_count + 1
                    
                    
                    # save stats
                    if(total_inst_found in size_of_BB):
                        index = size_of_BB.index(total_inst_found)
                        independent_instructions[index].append(independent_inst_found)
                        independent_instructions_waw_war[index].append(independent_inst_found_waw_war)
                        inst_with_WAW_WAR[index].append(inst_with_only_false_deps)
                        number_of_reorderings[index].append(num_reorderings)
                    else:
                        size_of_BB.append(total_inst_found)
                        independent_instructions.append([])
                        independent_instructions[-1].append(independent_inst_found)
                        independent_instructions_waw_war.append([])
                        independent_instructions_waw_war[-1].append(independent_inst_found_waw_war)
                        inst_with_WAW_WAR.append([])
                        inst_with_WAW_WAR[-1].append(inst_with_only_false_deps)
                        number_of_reorderings.append([])
                        number_of_reorderings[-1].append(num_reorderings)
                        
                        if(inst_with_only_false_deps):
                            print("**********************************")
                            print("#################FALSE DEPS INST")
                            for inst in false_deps_list:
                                print(inst)
                            print("#################BASIC BLOCK")
                            for inst in inst_list:
                                print(inst)
                    
                    #print("num_reorderings ",num_reorderings)
                    inst_list = []
                elif read_file:
                    inst_list.append(line)
                
                if("warp" in  line):
                    read_file = False
                if("inst" in line):
                    read_file = True
                    inst_list = []
                    
                line_count = line_count + 1
                
        for i in range(len(size_of_BB)):
            print("idx ",i," BBsize ",size_of_BB[i]," count ",len(inst_with_WAW_WAR[i]),end="")
            #for j in range(len(inst_with_WAW_WAR)):
                # if(size_of_BB[i] != 0):
                #     print(" war_waw ",inst_with_WAW_WAR[i][j]," avg ",inst_with_WAW_WAR[i][j]/size_of_BB[i]," ",end="")
            if(size_of_BB[i] != 0):
                avg_inst = sum(inst_with_WAW_WAR[i])/(len(inst_with_WAW_WAR[i])*size_of_BB[i])*100
                print(" avg false dependencies ",avg_inst,end="")
            print()       
                
                
        # total_BB = 0 
        # val = 0
        # total_avg_with_false = 0
        # for i in range(len(size_of_BB)):
        #     print("BB_size ",size_of_BB[i]," ",end="")
        #     if(size_of_BB[i]!=0):
        #         avg_indep = sum(independent_instructions[i]) / (len(independent_instructions[i])*size_of_BB[i]) # change
        #         avg_with_false = sum(inst_with_WAW_WAR[i]) / (len(inst_with_WAW_WAR[i]))
        #         total_avg_with_false = total_avg_with_false + avg_with_false
        #         for j in range(len(independent_instructions[i])):
        #             avg = independent_instructions[i][j] / size_of_BB[i]
        #             val = independent_instructions[i][j] + val 
        #             total_BB = total_BB + 1
        #     else:
        #         avg_indep = 0
        #         avg_with_false = 0
        #     avg_indep_war_waw = sum(independent_instructions_waw_war[i]) / len(independent_instructions_waw_war[i])
        #     avg_reorderings = sum(number_of_reorderings[i]) / len(number_of_reorderings[i])
        #     #print("avg_indep ",avg_indep," perc ",len(number_of_reorderings[i])/BB_count*100)
        #     print("Num inst with false deps ",avg_with_false," ",sum(inst_with_WAW_WAR[i])," ",(len(inst_with_WAW_WAR[i])))
        #     val = 0
        # #print("average percent independent instructions "," total_BB ",total_BB," final ",(avg_WAW_WAR/total_BB)*100)
        # print("***** Instructions with false dependencies ",)
        
def get_total_cycles(fin,filename,foldername):

    cycle_count = 0
    
    valid = 0
    
    for line in fin:
        line = line.strip()
        line1 =  line.split(' ')

        # if "TOTAL CYCLES " in line:
        #     cycle_count = float(line1[3])
        
        if(line1[0] == "gpu_tot_sim_cycle"):
            cycle_count = float(line1[2])
        
        # if("total_times_in_cycle" in line):
        #     cycle_count = float(line1[1])
            
        if("exit detected" in line):
            valid = 1
            
    
    if(valid != 1):
        cycle_count = 0   
        
    return cycle_count


def get_cycles_per_kernel(fin,filename,foldername):

    cycle_count = []
    kernel_name = []
    inst_count = []
    occupancy = []
    issued_cta = []
    
    valid = 0
    # get combined per kernel speedup 
    
    kernel_temp = "NULL"
    index = -1
    
    for line in fin:
        line = line.strip()
        line1 =  line.split(' ')
        
        if(line1[0] == "gpu_sim_cycle"):
            if(index == -1):
                cycle_count.append(float(line1[2]))
            else:
                cycle_count[index] = cycle_count[index] + float(line1[2])
        
        if("exit detected" in line):
            valid = 1
            
        if(line1[0] == "kernel_name"):
            kernel_temp = line1[2]
            if(kernel_temp in kernel_name):
                index = kernel_name.index(kernel_temp)
            else:
                kernel_name.append(line1[2])
                index = -1
            
        if(line1[0] == "gpu_sim_insn"):
            if(index == -1):
                inst_count.append(float(line1[2]))
            else:
                inst_count[index] = inst_count[index] + float(line1[2])
            
        if(line1[0] == "gpu_occupancy"):
            if(index == -1):
                occupancy.append((line1[2]))
            else:
                occupancy[index] = occupancy[index]
            
        if(line1[0] == "gpu_tot_issued_cta"):
            if(index == -1):
                issued_cta.append(float(line1[2]))
            else:
                issued_cta[index] = (issued_cta[index] + float(line1[2])) / 2
    
    if(valid != 1):
        cycle_count = []   
        
    return cycle_count, kernel_name, inst_count, occupancy, issued_cta

def get_OoO_inst_count(fin):
    
    tot_inst = 1
    OoO_inst = 0
    
    indep_inst_streams = []
    for i in range(32):
        indep_inst_streams.append(0)
    
    for line in fin:
        line = line.strip()
        line1 =  line.split(' ')
        
        if "tot_issues_OOO" in line:
            tot_inst = float(line1[1])
        if "stalled_inst_tot" in line:
            OoO_inst = float(line1[6])
           
        
            
    OoO_inst =   OoO_inst/ tot_inst * 100
    return OoO_inst
        

def get_l2_miss_rate(fin):

    cycle_count = 0

    for line in fin:
        line = line.strip()
        line1 =  line.split(' ')

        if "L2_total_cache_miss_rate " in line:
            cycle_count = float(line1[2])

    return cycle_count

def get_num_scheduled(csv,all_folders_stats,benchmarks,directory):

    for foldername in all_folders_stats:
        cycles = []
        #for filename in benchmarks:
        for filename in benchmarks:
            filepath = directory+"/"+foldername+"/"+filename
            fin=open(filepath,"r")
            indep_sched_counter, total_count = collect_num_sched(fin,filename)

def collect_num_sched(fin,filename):

    cycle_count = []
    print("IN_FILE ",filename)

    for line in fin:
        line = line.strip()
        line1 =  line.split(' ')
        if(line1[0] == "INST_ISSUED" and len(line1)>3):
            warp_id = int(line1[1])
            sid = int(line1[2])
            pc = int(line1[-1])

            # find all schedules
            while(len(cycle_count)<(sid+1)):
                cycle_count.append([])
            
            while(len(cycle_count[sid])<(warp_id+1)):
                cycle_count[sid].append([])

            if(pc == 0):
                cycle_count[sid][warp_id].append([])

            if(len(cycle_count[sid][warp_id])>0):
                cycle_count[sid][warp_id][-1].append(pc)
    
    # find indep schedules and get count
    total_count = 0
    indep_sched = []
    indep_sched_counter = []
    for sid in range(len(cycle_count)):
        for wid in range(len(cycle_count[sid])):
            for schedule_num in range(len(cycle_count[sid][wid])):
                sched = cycle_count[sid][wid][schedule_num]

                sched = cycle_count[sid][wid]

                if sched in indep_sched:
                    index = indep_sched.index(sched)
                    indep_sched_counter[index] += 1
                else:
                    indep_sched.append(sched)
                    indep_sched_counter.append(1)

                total_count += 1

    return indep_sched_counter, total_count

def get_all_stalls(fin):
        
    cycle_count = []
    for i in range(6):
        cycle_count.append(-1)

    for line in fin:
        line = line.strip()
        line1 =  line.split(' ')

        # if("cycle_found" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[0] = line1[1]
        # if("indep_mem_inst_stuck" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[1] = line1[1]
        # if("indep_comp_inst_stuck" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[2] = line1[1]
        # if("indep_inst_stuck" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[3] = line1[1]
        # if("indep_mem_inst_found" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[4] = line1[1]
        # if("indep_comp_inst_found" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[5] = line1[1]
        # if("indep_inst_found" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[6] = line1[1]
        # if("inst_issued_ooo_by_ghost" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[7] = line1[1]
        # if("stall_found" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[8] = line1[1]
        # if("indep_mem_inst_stuck_load_dep" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[9] = line1[1]
        # if("indep_comp_inst_stuck_load_dep" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[10] = line1[1]
        # if("indep_inst_stuck_load_dep" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[11] = line1[1]
        # if("indep_mem_inst_found_load_dep" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[12] = line1[1]
        # if("indep_comp_inst_found_load_dep" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[13] = line1[1]
        # if("indep_inst_found_load_dep" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[14] = line1[1]
        # if("load_dep_found" in line):
        #     #cycle_count.append(line1[1])
        #     cycle_count[15] = line1[1]

        if("cycle_found" in line):
            cycle_count[0] = line1[1]
        if("stall_found" in line):
            cycle_count[1] = line1[1]
        if("indep_inst_stuck" in line):
            cycle_count[2] = line1[1]
        if("tot_num_SM_cycles" in line):
            cycle_count[3] = line1[1]
        if("tot_num_SM_stall" in line):
            cycle_count[4] = line1[1]
        if("tot_num_indep_inst_found_on_stall_SM" in line):
            cycle_count[5] = line1[1]

    #print(cycle_count)

    return cycle_count

def get_all_stats(fin, filename):

    # get average occupancy of benchmarks

    occupancy = 1
    ipc = 0
    mlp = 0

    for line in fin:
        line = line.strip()
        line1 =  line.split(' ')

        if(line1[0] == "gpu_tot_occupancy"):
            occupancy = (line1[2][:-1])
        
        # if(line1[0] == "gpu_tot_sim_insn"):
        #     occupancy = (line1[-1])
        
        if("limited by" in line):
            ipc = line1[-1]
            
        # if(line1[0] == "gpu_tot_sim_cycle"):
        #     ipc = float((line1[-1]))
        
        # if(line1[0] == "tot_inst_exec"):
        #     ipc = float(line1[1])
            
        # if(line1[0] == "tot_issues_OOO"):
        #     mlp = float((line1[-1]))
            
        # if(line1[0] == "total_sched_cycles"):
        #     total_sched_cycles = float((line1[1]))
        #     if(total_sched_cycles > 0):
        #         tot_inst_ret = float((line1[3]))  
        #         ipc = tot_inst_ret / total_sched_cycles * 100
        #     else:
        #         ipc = -2
        
        # if(line1[0] == "gpu_tot_max_power"):
        #     ipc = float((line1[-1]))
            
        # if(line1[0] == "gpu_tot_avg_power"):
        #     ipc = float((line1[-1]))
        
        # if(line1[0] == "gpu_tot_sim_insn"):
        #     ipc = float((line1[-1]))

        # if(line1[0] == "STALLING_VALUES"):
        #     mlp = float((line1[1]))
        
        
        # if(line1[0] == "total_sched_cycles"):
        #     ipc = float((line1[1]))
        #     mlp = float((line1[2])) + float((line1[3]))
        
        # if(line1[0] == "total_times_in_cycle"):
        #     occupancy = float((line1[1]))
            
        # if(line1[0] == "STALLING_VALUES"):
        #     ipc = float((line1[1]))
        
    # if(occupancy!=0):
    #     occupancy = ipc / occupancy*100
    
    return occupancy, ipc, mlp

def get_stalling_inst(csv,all_folders,all_bmrks,directory):
    cycles_overall = []
    for foldername in all_folders:
        cycles = []
        #for filename in benchmarks:
        for filename in all_bmrks:
            filepath = directory+"/"+foldername+"/"+filename
            fin=open(filepath,"r")
            cycle_count = get_all_stalls(fin)
            cycles.append(cycle_count)
        cycles_overall.append(cycles)

    os.chdir(directory)

    csv.write("Benchmark,cycle_found,indep_mem_inst_stuck,indep_comp_inst_stuck,indep_inst_stuck,indep_mem_inst_found,indep_comp_inst_found,indep_inst_found,inst_issued_ooo_by_ghost,stall_found,indep_mem_inst_stuck_load_dep,indep_comp_inst_stuck_load_dep,indep_inst_stuck_load_dep,indep_mem_inst_found_load_dep,indep_comp_inst_found_load_dep,indep_inst_found_load_dep,load_dep_found\n")

    csv_line = []
    # this has not been scaled
    for i in range(len(cycles)):
        csv_line = []
        csv_line.append(all_bmrks[i])
        length = min(16,len(cycles_overall[0][i]))
        for j in range(length):
            csv_line.append(cycles_overall[0][i][j])
        joined_string = ','.join(map(str, csv_line))
        csv.write(joined_string)
        csv.write("\n")


def get_speedup_numbers(csv,all_folders,all_bmrks,directory):
    cycles_overall = []
    for foldername in all_folders:
        cycles = []
        #for filename in benchmarks:
        for filename in all_bmrks:
            filepath = directory+"/"+foldername+"/"+filename
            fin=open(filepath,"r")
            cycle_count = get_total_cycles(fin,filename,foldername)
            cycles.append(cycle_count)
        cycles_overall.append(cycles)

    all_folders1 = ["IBOOO_8"]
    
    # # get occupancy data
    cycles_occupancy = []
    cycles_ipc = []
    cycles_mlp = []
    for foldername in all_folders1:
        occupancy_list = []
        ipc_list = []
        mlp_list = []
        #for filename in benchmarks:
        for filename in all_bmrks:
            filepath = directory+"/"+foldername+"/"+filename
            fin=open(filepath,"r")
            occupancy, ipc, mlp = get_all_stats(fin, filename)
            occupancy_list.append(occupancy)
            ipc_list.append(ipc)
            mlp_list.append(mlp)
        cycles_occupancy.append(occupancy_list)
        cycles_ipc.append(ipc_list)
        cycles_mlp.append(mlp_list)

    os.chdir(directory)

    line = "Benchmark,"
    for bmrk in all_folders[1:]:
        line = line + bmrk + ","
    # for bmrk in all_folders1:
    #     line =  line + bmrk+"_OoOperc,"
    for bmrk in all_folders1:
        line =  line + bmrk+"_occupancy,"
    for bmrk in all_folders1:
        line =  line + bmrk+"_limitedBy,"
    
    # for bmrk in all_folders:
    #     line =  line + bmrk+"total_times_in_cycle,"
    # for bmrk in all_folders:
    #     line =  line + bmrk+"_schedStall,"
    
    # for bmrk in all_folders:
    #     line =  line + bmrk+"_insn,"
    # for bmrk in all_folders:
    #     line =  line + bmrk+"_per_OOO,"
    
    line = line + "\n"
    
    csv.write(line)

    speedup_overall = []
    # done for scaling
    for i in range(len(all_folders)-1):
        speedup = []
        for j in range(len(cycles)):
            if(cycles_overall[0][j] > 0):
                speedup_val = (cycles_overall[0][j] - cycles_overall[i+1][j]) / cycles_overall[0][j] + 1
            else:
                speedup_val = 0
            speedup.append(speedup_val)
        speedup_overall.append(speedup)

    csv_line = []

    # scaled code
    for i in range(len(cycles)):
        csv_line = []
        #if(cycles_overall[1][i]!=0):
        csv_line.append(all_bmrks[i])
        #csv_line.append(cycles_overall[0][i])
        for j in range(len(all_folders)-1):
            #csv_line.append(cycles_overall[j+1][i])
            csv_line.append(speedup_overall[j][i])
        for j in range(len(all_folders1)):
            csv_line.append(cycles_occupancy[j][i])
        for j in range(len(all_folders1)):
            csv_line.append(cycles_ipc[j][i])
        # length = min(6,len(cycles_overall_stats[0][i]))
        # for j in range(length):
        #    csv_line.append(cycles_overall_stats[0][i][j])
        # csv_line.append(cycles_occupancy[0][i])
        # csv_line.append(cycles_ipc[0][i])
        # csv_line.append(cycles_mlp[0][i])
        
        
        # for j in range(len(all_folders)):
        #     csv_line.append(cycles_occupancy[j][i])
        # for j in range(len(all_folders)):
        #     csv_line.append(cycles_ipc[j][i])
        # for j in range(len(all_folders)):
        #     csv_line.append(cycles_mlp[j][i])
        
        joined_string = ','.join(map(str, csv_line))
        csv.write(joined_string)
        csv.write("\n")
        
def get_per_kernel_speedup(csv,all_folders,filename,directory):
    cycles_overall = []
    kernel_name_overall = []
    inst_count_overall = []
    inst_occupancy_overall = []
    occupancy_overall = []
    issued_cta_overall = []
    for foldername in all_folders:
        filepath = directory+"/"+foldername+"/"+filename
        fin=open(filepath,"r")
        cycle_count,kernel_name,num_inst,occupancy,issued_cta = get_cycles_per_kernel(fin,filename,foldername)
        cycles_overall.append(cycle_count)
        for i in range(len(kernel_name)):
            kernel_name[i] = kernel_name[i]+"_"+filename
        kernel_name_overall.append(kernel_name)
        inst_count_overall.append(num_inst)
        occupancy_overall.append(occupancy)
        issued_cta_overall.append(issued_cta)
        
    os.chdir(directory)
    
    # line = "Benchmark,"
    # for bmrk in all_folders[1:]:
    #     line = line + bmrk + ","
    # line = line + "NumCycles,"
    # line = line + "Occupancy,"
    # line = line + "issued_cta,"
    # line = line + "\n"
    
    # csv.write(line)

    speedup_overall = []
    # done for scaling
    for i in range(len(all_folders)-1):
        speedup = []
        for j in range(len(cycles_overall[0])):
            if(cycles_overall[0][j] > 0):
                speedup_val = (cycles_overall[0][j] - cycles_overall[i+1][j]) / cycles_overall[0][j] + 1
            else:
                speedup_val = 0
            speedup.append(speedup_val)
        speedup_overall.append(speedup)

    csv_line = []

    # scaled code
    for i in range(len(cycles_overall)-1):
        assert(len(cycles_overall[0])==len(cycles_overall[i+1]))    
    
    for i in range(len(cycles_overall[0])):
        csv_line = []
        csv_line.append(kernel_name_overall[0][i])
        for j in range(len(all_folders)-1):
            csv_line.append(speedup_overall[j][i])
        for j in range(len(all_folders)-1):
            csv_line.append(inst_count_overall[0][i])
        for j in range(len(all_folders)-1):
            csv_line.append(occupancy_overall[0][i])
        for j in range(len(all_folders)-1):
            if(i == 0):
                csv_line.append(issued_cta_overall[0][i])
            else:
                csv_line.append((issued_cta_overall[0][i]-issued_cta_overall[0][i-1]))
        
        joined_string = ','.join(map(str, csv_line))
        csv.write(joined_string)
        csv.write("\n")
        

def get_load_distribution(benchmark,folder,directory):
    PCs = [976,1440,1232,992,1472,1248]
    for foldername in folder:
        for filename in benchmark:
            PC_vals_temp = []
            # add a list for each PC
            for i in range(len(PCs)):
                PC_vals_temp.append([])
            filepath = directory+"/"+foldername+"/"+filename
            fin=open(filepath,"r")
            collect_numbers = False
            for line in fin:
                line = line.strip()
                line1 =  line.split(' ')
                if("kernel id" in line):
                    if(int(line1[-1])==3):
                        collect_numbers = True
                    else:
                        collect_numbers = False
                
                if "INST_ISSUE_CYCLE" in line and collect_numbers == True:
                    PC = int(line1[1])
                    num_cycle = int(line1[3])
                    if PC in PCs:
                        index = PCs.index(PC)
                        PC_vals_temp[index].append(num_cycle)
            
            #print vals
            for i in range(len(PCs)):
                word = "val_"+str(PCs[i])+" = "
                print(word,end="")
                print(PC_vals_temp[i])
                    
    
def get_OoO_inst(csv,all_folders,all_bmrks,directory):
    per_OoO_inst_tot = []
    for foldername in all_folders:
        per_OoO_inst = []
        #for filename in benchmarks:
        for filename in all_bmrks:
            filepath = directory+"/"+foldername+"/"+filename
            fin=open(filepath,"r")
            OoO = get_OoO_inst_count(fin)
            per_OoO_inst.append(OoO)
        per_OoO_inst_tot.append(per_OoO_inst)
        
        
    os.chdir(directory)   
    line = "Benchmark,"
    for bmrk in all_folders[1:]:
        line = line + bmrk + ","
    
    line = line + "\n"
    csv.write(line)
        
    csv_line = []
    
    for i in range(len(all_bmrks)):
        csv_line = []
        csv_line.append(all_bmrks[i])
        csv_line.append(per_OoO_inst_tot[0][i])
        #csv_line.append(per_OoO_inst_tot[1][i])
        
        joined_string = ','.join(map(str, csv_line))
        csv.write(joined_string)
        csv.write("\n")
        
        
def get_delay_stats(csv,all_folders,all_bmrks,directory):
    
    PC_overall = []
    cycles_overall = []
    sd_overall = []
    mean_overall = []
    for foldername in all_folders:
        for filename in all_bmrks:
            filepath = directory+"/"+foldername+"/"+filename
            fin=open(filepath,"r")
            cycles_hist  = []
            cycles_list = []
            op_list = []
            tot_inst = 0
            PC = []
            cycles = []
            op = []
            sd = []
            mean = []
            max_val = []
            min_val = []
            
            for line in fin:
                line = line.strip()
                line1 =  line.split(' ')

                if "INST_ISSUE_CYCLE " in line:
                    PC_val = float(line1[1])
                    cycles_val = float(line1[3])
                    op_val = float(line1[2])
                    tot_inst = tot_inst + 1
                    
                    if(PC_val in PC):
                        loc = PC.index(PC_val)
                        cycles[loc].append(cycles_val)
                        op[loc].append(op_val)
                    else:
                        PC.append(PC_val)
                        cycles.append([])
                        cycles[-1].append(cycles_val)
                        op.append([])
                        op[-1].append(op_val)
                        
            for i in range(len(PC)):
                res = statistics.pstdev(cycles[i])
                sd.append(res)
                res = np.average(cycles[i])
                mean.append(res)
                min_val.append(min(cycles[i]))
                max_val.append(max(cycles[i]))
                
            # reoder lists
            total_inst = []
            op_final = []
            for i in range(len(PC)):
                total_inst.append(len(cycles[i]))
                op_final.append(op[i][0])
                
            print("***********BENCHMARK VALUES ",filepath)
            for i in range(len(PC)):
                if(sd[i]!=0 and op_final[i]!=100):
                    if(mean[i]==0):
                        mean[i]=1
                    print("i ",i,"PC_VAL ",(int(PC[i])),":",hex(int(PC[i]))," op ",op_final[i]," sd ",round(sd[i],2)," mean ",round(mean[i],2)," div ",round(sd[i]/mean[i],2), " val ",total_inst[i]," max ",max_val[i]," min ",min_val[i]," perc_tot ",round((total_inst[i]/tot_inst*100),2))

def get_reorderings_info(csv,fin,directory):

    # store the values of PCs and how much time they take for each warp
    warp_level_data = []
    for i in range(100):
        warp_level_data.append({})

    kernel_list = []

    csv.write("PC,-5,-4,-3,-2,-2,0,1,2,3,4,5\n")

    # keep per inst data, dont think about keep warp level info
    inst_level_data = {}
    inst_level_data_per_warps = {}
    inst_level_data_long_op = {}
    inst_level_data_stalls = {}
    inst_level_data_stalls_per_dist = {}
    inst_level_data_instruction_keeper = {}
    inst_level_data_in_IB = {}
    
    for line in fin:
        line = line.strip()
        line1 =  line.split(' ')

        if("VALUE_REORDERING" in line):
            dist = int(line1[2])
            pc = int(line1[1])
            warp = int(line1[3])
            kernel = line1[4]  
            stalls = int(line1[5])
            long_op = int(line1[6])
            #in_IB = int(line1[7])
            in_IB = int(line1[6])

            ## add warp level stats
            if(pc in warp_level_data[warp]):
                warp_level_data[warp][pc].append(dist)
            else:
                warp_level_data[warp].setdefault(pc, [])
                warp_level_data[warp][pc].append(dist)                
                    
            # add stats
            if(kernel not in inst_level_data):    
                inst_level_data.setdefault(kernel, {})
                inst_level_data_long_op.setdefault(kernel, {})
                inst_level_data_stalls.setdefault(kernel, {})
                inst_level_data_in_IB.setdefault(kernel, {})
                inst_level_data_stalls_per_dist.setdefault(kernel, {})
                inst_level_data_per_warps.setdefault(kernel, {})
                inst_level_data_instruction_keeper.setdefault(kernel, {})
            if(warp not in inst_level_data_per_warps[kernel]):
                inst_level_data_per_warps[kernel].setdefault(warp, {})
            if(pc not in inst_level_data[kernel]):
                inst_level_data[kernel].setdefault(pc, [])
                inst_level_data_stalls[kernel].setdefault(pc, [])
                inst_level_data_in_IB[kernel].setdefault(pc, [])
                inst_level_data_stalls_per_dist[kernel].setdefault(pc, {})
                inst_level_data_long_op[kernel].setdefault(pc, long_op)
                inst_level_data_instruction_keeper[kernel].setdefault(pc,line)
            if(pc not in inst_level_data_per_warps[kernel][warp]):
                inst_level_data_per_warps[kernel][warp].setdefault(pc, [])
            if(dist not in inst_level_data_stalls_per_dist[kernel][pc]):
                inst_level_data_stalls_per_dist[kernel][pc].setdefault(dist, {})
                inst_level_data_stalls_per_dist[kernel][pc][dist][stalls] = 0
            if(stalls not in inst_level_data_stalls_per_dist[kernel][pc][dist]):
                inst_level_data_stalls_per_dist[kernel][pc][dist][stalls] = 0
            inst_level_data[kernel][pc].append(dist)
            inst_level_data_stalls[kernel][pc].append(stalls)
            inst_level_data_in_IB[kernel][pc].append(in_IB)
            inst_level_data_stalls_per_dist[kernel][pc][dist][stalls] += 1
            inst_level_data_per_warps[kernel][warp][pc].append(dist)

    values_to_search = [-5,-4,-3,-2,-1,0,1,2,3,4,5]

    var_reorder = []
    var_dist = []
    var_quantity = []

    box_reorder = []
    box_dist = []
    pc_list = []

    counter1 = 0
    
    # SASS folder 
    SASS_folder = "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run/traces/device-0/11.5/lonestar-bfs/__data_rmat12_sym_gr/traces"
    files_to_read = [
        "kernel-2.traceg",
    ]
    files_to_write = [
       "ignore_file",
    ]
    
    final_list_to_look_up = {}
        
    # # get the compiler data
    for key in  inst_level_data:
        final_list_to_look_up.setdefault(key, {})
        for pc in inst_level_data[key]:
            final_list_to_look_up[key].setdefault(pc, {})
            stats = []
            dist = []
            tot_values = 0
            for val in inst_level_data[key][pc]:
                tot_stall_dist = 0
                if val in dist:
                    idx = dist.index(val)
                    stats[idx]+=1
                else:
                    stats.append(1)
                    dist.append(val)
                tot_values += 1

            dist = [x for _,x in sorted(zip(stats,dist), reverse=True)]
            stats = [x for x in sorted((stats), reverse=True)]
            
            final_list_to_look_up[key][pc]["distance"] =  dist
            final_list_to_look_up[key][pc]["stats"] =  stats
            
    # # read from one file and write to new file with modified schedule
    
    os.chdir(SASS_folder)
    
    # get the pc list from here
    pc_list_here = {}
    
    for file_num in range(len(files_to_read)):
        reading_file = files_to_read[file_num]
        readf=open(reading_file,"r")
        
        start_reading = False
        
        for line in readf:
            line = line.strip()
            line1 =  line.split(' ')
            
            if("kernel name" in line):
                kernel =  line1[-1]
                if(kernel not in pc_list_here):
                    pc_list_here.setdefault(kernel, {})
            
            # start reading the lines and reordering
            if(line1[0]=="insts"):
                start_reading = True
            
            if(start_reading == True and len(line1)>1):
                pc = line1[0]
                pc_val = pc[0]
                if(pc_val.isdigit()==True):
                    pc1 = int(pc, 16)
                    if(pc1 not in pc_list_here[kernel]):
                        pc_list_here[kernel].setdefault(pc1,line)
                        
    # for kenrel in pc_list_here:
    #     for pc in pc_list_here[kernel]:
    #         print("VAL ",pc," ",pc_list_here[kernel])
                    
    for file_num in range(len(files_to_write)):
        reading_file = files_to_read[file_num]
        writing_file = files_to_write[file_num]
        readf=open(reading_file,"r")
        writef=open(writing_file,"w")
        start_reading = False
        
        final_list_to_put = []
        
        for line in readf:
            line = line.strip()
            line1 =  line.split(' ')

            # get kernel we want to reorder
            if("kernel name" in line):
                kernel =  line1[-1]
                
            # start reading the lines and reordering
            if(line1[0]=="insts"):
                start_reading = True
            
            if(start_reading == True and len(line1)>1):
                pc = line1[0]
                pc_val = pc[0]
                if(pc_val.isdigit()==True):
                    pc1 = int(pc, 16)
                    index = final_list_to_look_up[key][pc1]["stats"].index(max(final_list_to_look_up[key][pc1]["stats"]))
                    # print("PC_put ",hex(pc1)," dist ",final_list_to_look_up[kernel][pc1]["distance"]," stats ",final_list_to_look_up[key][pc1]["stats"]," index ",index)
                    distance = final_list_to_look_up[kernel][pc1]["distance"][index]
                    if(distance>=0):
                        final_list_to_put.append(line)
                    else:
                        final_list_to_put.insert(distance,line)   
                else:
                    final_list_to_put.append(line)         
            else:
                final_list_to_put.append(line)
        for line in final_list_to_put:
            line = line+'\n'
            writef.write(line)
    
    os.chdir(directory)
            
            
    # large prints
    for key in  inst_level_data:
        for pc in inst_level_data[key]:
            stats = []
            dist = []
            stalls_type = []
            stalls_counter = []
            tot_values = 0
            ib_type = []
            ib_counter = []
            for val in inst_level_data[key][pc]:
                tot_stall_dist = 0
                if val in dist:
                    idx = dist.index(val)
                    stats[idx]+=1
                else:
                    stats.append(1)
                    dist.append(val)
                tot_values += 1
            for val in inst_level_data_stalls[key][pc]:
                if val in stalls_type:
                    idx = stalls_type.index(val)
                    stalls_counter[idx]+=1
                else:
                    stalls_counter.append(1)
                    stalls_type.append(val)
                    
            for val in inst_level_data_in_IB[key][pc]:
                if val in ib_type:
                    idx = ib_type.index(val)
                    ib_counter[idx]+=1
                else:
                    ib_counter.append(1)
                    ib_type.append(val)

            # get the variance on OoO issue
            var_reorder.append(statistics.pvariance(inst_level_data[key][pc]))
            var_dist.append(statistics.pvariance(inst_level_data_stalls[key][pc]))
            var_quantity.append(len(inst_level_data_stalls[key][pc]))

            box_reorder.append(inst_level_data[key][pc])
            box_dist.append(inst_level_data_stalls[key][pc])
            #print(statistics.pvariance(inst_level_data_stalls[key][pc])," VAL ",inst_level_data_stalls[key][pc])
            pc_list.append(counter1)
            counter1 += 1

            # print all data for analysis
            print("*****************************")
            stalls_type = [x for _,x in sorted(zip(stalls_counter,stalls_type), reverse=True)]
            stalls_counter = [x for x in sorted((stalls_counter), reverse=True)]
            
            dist = [x for _,x in sorted(zip(stats,dist), reverse=True)]
            stats = [x for x in sorted((stats), reverse=True)]
            print("PC_ACTUAL ",pc_list_here[key][pc])
            if(inst_level_data_long_op[key][pc] == 1):
                print_all = 1
                print("MEM_KERNEL ",key," pc: ",hex(pc)," ",pc," TOTAL ",tot_values," --> ",end="") 
                for i in range(len(stats)):
                    print(" dis ",round(dist[i],2)," : val ",round(stats[i],2)," per ",round(((stats[i])/(float)(tot_values)*100),2)," ; ",end="") 
                #     if(((stats[i])/(float)(tot_values)*100)!=100):
                #         print(" RANGE: ",inst_level_data_stalls_per_dist[key][pc][dist[i]]," || ",end="")
                #     else:
                #         print_all = 0
                # if(print_all == 1):
                # print(" LATENCY || ")
                # for i in range(len(stalls_type)):
                #     if(((float)(stalls_counter[i])/(float)(tot_values)*100)>1):
                #         print(stalls_type[i]," : val ",stalls_counter[i]," per ",((float)(stalls_counter[i])/(float)(tot_values)*100)," || ",end="") 
                # print(" TIME SPENT || ")
                # for i in range(len(ib_type)):
                #     if(((float)(ib_counter[i])/(float)(tot_values)*100)>1):
                #         print(ib_type[i]," : val ",ib_counter[i]," per ",((float)(ib_counter[i])/(float)(tot_values)*100)," || ",end="") 
            else:
                print_all = 1
                print("NON_KERNEL ",key," pc: ",hex(pc)," ",pc," TOTAL ",tot_values," --> ",end="") 
                for i in range(len(stats)):
                    print(" dis ",round(dist[i],2)," : val ",round(stats[i],2)," per ",round(((float)(stats[i])/(float)(tot_values)*100),2)," ; ",end="")
                #     if(((stats[i])/(float)(tot_values)*100)!=100):
                #         print(" RANGE: ",inst_level_data_stalls_per_dist[key][pc][dist[i]]," || ",end="")
                #     else:
                #         print_all = 0
                # if(print_all == 1):
                # print("\n LATENCY || ")
                # for i in range(len(stalls_type)):
                #     if(((float)(stalls_counter[i])/(float)(tot_values)*100)>1):
                #         print(stalls_type[i]," : val ",stalls_counter[i]," per ",((float)(stalls_counter[i])/(float)(tot_values)*100)," || ",end="")
                # print("\n TIME SPENT || ")
                # for i in range(len(ib_type)):
                #     if(((float)(ib_counter[i])/(float)(tot_values)*100)>1):
                #         print(ib_type[i]," : val ",ib_counter[i]," per ",((float)(ib_counter[i])/(float)(tot_values)*100)," || ",end="") 
            print()

    # get the variances to plot
    #print(var_reorder)
    #print(var_dist)  
    # print(pc_list)
    # print(var_quantity)

    # # get box plot data
    # print(box_reorder)
    # # print(box_dist)
    # print(pc_list)
    # print(var_quantity)

def get_memory_dep_reorderings(csv,all_folders,benchmarks,directory):

    all_folders = ["IBOOO_8_complete_run"]
    benchmarks = ["out_ooo"]
    
    cycles_overall = []
    for foldername in all_folders:
        cycles = []
        #for filename in benchmarks:
        for filename in benchmarks:
            filepath = directory+"/"+foldername+"/"+filename
            fin=open(filepath,"r")
            get_reorderings_info(csv,fin,directory)
            
def get_compiler_stats(file):

    fin=open(file,"r")
    #line_to_put_in = "0fa0 ffffffff 1 R8 IADD3 2 R8 R255 0"
    #line_to_put_after = "0f60 ffffffff 1 R9 FFMA 3 R12 R14 R17 0"

    #line_to_put_in = "0fa0 ffffffff 1 R8 IADD3 2 R8 R255 0"
    #line_to_put_after = "0f50 ffffffff 1 R14 LDG.E.SYS 1 R6 4 1"

    line_to_put_after1 = "0f20 ffffffff 1 R17 FFMA 3 R12 R14 R13 0"
    line_to_put_in1 = "0fa0 ffffffff 1 R8 IADD3 2 R8 R255 0"

    line_to_put_after2 = "0f60 ffffffff 1 R9 FFMA 3 R12 R14 R17 0"
    line_to_put_in2 = "0fb0 ffffffff 0 ISETP.NE.AND 2 R8 R255 0"

    for line in fin:
        line = line.strip()
        line1 =  line.split(' ')

        # make sure we skip the line we inserting OoO manually'
        if(line_to_put_in1 not in line and line_to_put_in2 not in line):
            
            #if we have line after which we need to add the line to skip
            if(line_to_put_after2 not in line and line_to_put_after1 not in line):
                print(line)
            else:
                if(line_to_put_after2 in line):
                    print(line)
                    print(line_to_put_in2)
                if(line_to_put_after1 in line):
                    print(line)
                    print(line_to_put_in1)


def check_if_files_same(FILES,FOLDERS,directory):
    for filename in FILES:
        file_diff = []
        for foldername in FOLDERS:
            filepath = directory+"/"+foldername+"/"+filename
            fin=open(filepath,"r")
            filedata = []
            for line in fin:
                line = line.strip()
                filedata.append(line)
            file_diff.append(filedata)
        for i in range(len(file_diff)-1):
            for j in range(len(file_diff[0])):
                if(file_diff[0][j]!=file_diff[i+1][j]):
                    print("NOT EQUAL ",filename)

# Main Function
def main():
    
    all_folders = [
        "IN_2",
        "IBOOO_2",
        "IBOOO_4",
        "IBOOO_8",
        "IBOOO_16",
        "IBOOO_32",
        "IBOOO_64",
        "LOOG_OoO",
        "IBOOO_64_BP_RMI_RR_FINAL",
        "IBOOO_64_BP_RR_FINAL",
        "IBOOO_64_BP_RMI",
        "IBOOO_64_RMI_RR_FINAL",
        "IBOOO_64_RR_FINAL",
        "IBOOO_64_BP",
        "IBOOO_64_RMI",
        "IBOOO_4_free_on_oldest",
        "IBOOO_8_free_on_oldest",
        "IBOOO_16_free_on_oldest",
        "IBOOO_32_free_on_oldest",
        "IBOOO_64_free_on_oldest",
        "IBOOO_8_Reg",
        "IBOOO_8_BP",
        "IBOOO_8_RMI",
        "IBOOO_8_BP_RMI_RR",
    ]

    cycles_overall = []

    outfile = "cycle_count.csv"
    csv = open(outfile, "w")

    # benchmarks = []
    directory = os.getcwd()

    benchmarks = [ "slurm-backprop.out",
        #"slurm-bfs-rodinia.out", REMOVED
        #"slurm-bfs.out", REMOVED
        "slurm-bfs_rodinia.out",
        #"slurm-b+tree.out", REMOVED
        "slurm-b+tree_rodinia.out",
        "slurm-dwt2d_rodinia.out",
        #"slurm-dwt2d.out", REMOVED
        "slurm-gaussian.out",
        #"slurm-heartwall_rodinia.out",
        #"slurm-hotspot.out",
        #"slurm-hotspot_rodinia.out", REMOVED
        #"slurm-hotspot3D_rodinia.out", REMOVED
        "slurm-lavaMD_rodinia.out",
        "slurm-lud.out",
        "slurm-myocyte.out",
        "slurm-nn.out",
        #"slurm-nw.out", REMOVED
        #"slurm-nw_rodinia.out", REMOVED
        "slurm-particlefinder_float.out",
        #"slurm-pathfinder.out", REMOVED
        #"slurm-pathfinder_rodinia.out", REMOVED
        "slurm-srad_v1_rodinia.out",
        #"slurm-streamcluster_rodinia.out",
        "slurm-beamformer.out",
        "slurm-convolution_reduced.out",
        "slurm-dct.out",
        #"slurm-dct_pagoda.out", REMOVED
        #"slurm-des.out",REMOVED
        "slurm-des_pagoda.out",
        #"slurm-filterbank.out", REMOVED
        "slurm-mandelbort.out",
        #"slurm-matrixMul.out", REMOVED
        "slurm-matrixMul_pagoda.out",
        "slurm-multiwork.out",
        #"slurm-multiwork_pagoda.out", REMOVED
        "slurm-fw.out",
        #"slurm-fw_pannotia.out", REMOVED
        "slurm-sssp_pannotia.out",
        #"slurm-AES.out", CRASHES
        "slurm-ispass-BFS.out",
        "slurm-LPS.out",
        "slurm-LIB.out",
        #"slurm-MUM.out", CRASHES
        #"slurm-NQU.out",CRASHES
        "slurm-RAY.out",
        "slurm-STO.out",
        "slurm-CN.out",
        "slurm-GRU.out",
        "slurm-LSTM.out",
        # DEEPBENCH
        #"slurm-conv_bench.out",
        #"slurm-conv_bench-tencore.out",
        #"slurm-gemm_bench.out",
        #"slurm-gemm_bench-tencore.out",
        # "slurm-rnn_bench.out",
        # "slurm-rnn_bench-tencore.out",
    ]

    get_speedup_numbers(csv,all_folders,benchmarks,directory)
    
    # benchmark = ["sssp_load_data"]
    # folder = ["IN_2"]
    # get_load_distribution(benchmark,folder,directory)
    
    #all_folders = ["IBOOO_4_SIB_2_cycles"]
    
    # get the % of inst issued OoO
    #get_OoO_inst(csv,all_folders,benchmarks,directory)
    
    # all_folders = [
    #     "IBOOO_8_complete_run",
    # ]
    # benchmarks = ["out_ooo"]
    # # #benchmarks = ["mf_in"]
    # # get variablilty of instructions -> where OoO can benefit
    # get_delay_stats(csv,all_folders,benchmarks,directory)

    # benchmarks = ["out_ooo"]
    # # # generate the compiler optimized schedule based on GhOST
    # get_memory_dep_reorderings(csv,all_folders,benchmarks,directory)

    #get_stalling_inst(csv,all_folders_stats,benchmarks,directory)

    #get_num_scheduled(csv,all_folders_stats,benchmarks,directory)

    #file = "/scratch/gpfs/ishitac/gpgpusim-codes/accel-sim-framework/hw_run_back/polybench/11.0/polybench-bicg/NO_ARGS/traces/kernel-2-compiler-2.traceg"
    # generate the compiler file yourself
    #get_compiler_stats(file)

    # Generate a dependence graph for each basic block : 
    # 1. get the number of instructions without WAW and WAR dependencies in each baslic block
    # 2. Find the possible number of reorderings without renaming
    # 3. Find the possible number of reorderings with renaming
    # 4. Also find the # of indepndent instruction chains (any instruction with no dependence true and false is basically an independent instruction chain.)
    
    # Renaming_on = False
    # print("****************************")
    # print("Renaming OFF")
    # get_WAW_WAR_stats(Renaming_on)
    # Renaming_on = True
    # print("****************************")
    # print("Renaming ON")
    # get_WAW_WAR_stats(Renaming_on)
    
    # Remove false dependencies and put it in a new SASS file
    #remove_false_dependencies()
    
    # get per kernel speedup for kernel
    # all_folders = [
    #     "IN_2",
    #     "IBOOO_8",
    #     "IBOOO_8_no_mem_reordering",
    #     "IBOOO_8_MEM"
    # ]
    
    # line = "Benchmark,"
    # for bmrk in all_folders[1:]:
    #     line = line + bmrk + ","
    # line = line + "NumCycles,"
    # line = line + "Occupancy,"
    # line = line + "issued_cta,"
    # line = line + "\n"
    
    # csv.write(line)
    
    # for bmkr_of_interest in benchmarks:
    #     print("LOOKING AT ",bmkr_of_interest)
    #     get_per_kernel_speedup(csv,all_folders,bmkr_of_interest,directory)
    
    
    # FILES=["mandelbort.sh",
    # "bfs-rodinia.sh",
    # "bfs.sh",
    # "MaxFlops.sh",
    # "particlefilter_float_altis.sh",
    # "pagerank.sh",
    # "nn.sh",
    # "des.sh",
    # "ispass-BFS.sh",
    # "gaussian.sh",
    # "lonestar-bfs.sh",
    # "parboil-sgemm.sh",
    # "lonestar-mst.sh",
    # "dct.sh",
    # "nw-altis.sh",
    # "LPS.sh",
    # "GRU.sh",
    # "LSTM.sh",
    # "lud.sh",
    # "NQU.sh",
    # "STO.sh",
    # "RAY.sh",
    # "pannotia-bc.sh",
    # "dwt2d.sh",
    # "convbench.sh",
    # "parboil-mri-q.sh",
    # "parboil-cutcp.sh",
    # "LIB.sh",
    # "ispass-NN.sh",
    # "myocyte.sh",
    # "multiwork.sh",
    # "matrixMul.sh",
    # "parboil-stencil.sh",
    # "lonestar-sssp.sh",
    # "hotspot.sh",
    # "gemm_convbench.sh",
    # "rnn_bench.sh",
    # "CN.sh",
    # "filterbank.sh",
    # "particlefinder_float.sh",
    # "b+tree.sh",
    # "pathfinder.sh",
    # "fw.sh",
    # "beamformer.sh",
    # "srad.sh",
    # "convolution.sh",
    # "convbench_tencore.sh",
    # "lavaMD.sh",
    # "SN.sh",
    # "gups.sh",
    # "cutlass.sh",
    # "kmeans-altis.sh"]
    
    # FOLDER = ["IN_2","IBOOO_8"]
    
    # check_if_files_same(FILES,FOLDER,directory)



if __name__ == "__main__":
    main()