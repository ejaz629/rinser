# ida-analysis.py
# a simple IDAPython binary analysis script
# invoke with:
#   idat64 -c -A -S"ida-analysis.py $HOME/analysis.txt" <file.bin>

import sys
import idc
import idautils

f = open(idc.ARGV[1], 'a') if len(idc.ARGV) > 1 else sys.stdout
log = f.write

instr_count = 0
func_count = 0

# log current file path
log(idc.get_input_file_path() + '\n')

# wait for auto-analysis to complete
idc.auto_wait()

# count functions
log( 'count %d\n' % len(list(idautils.Functions())) )


for segea in Segments():
    for funcea in Functions(segea, get_segm_end(segea)):
        functionName = get_func_name(funcea)
        func_count += 1
        for (startea, endea) in Chunks(funcea):
            for head in Heads(startea, endea):
                log(functionName+ ' ' + ":"+ ' ' + "0x%08x"%(head)+ ' ' + ":"+ ' ' + GetDisasm(head)+ '\n')
                instr_count += 1

log(str(func_count))
log(str(instr_count))

# if logging to a file, close it and exit IDA Pro
if f != sys.stdout:
    f.close()
    idc.qexit(0)

