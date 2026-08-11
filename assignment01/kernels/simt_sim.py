"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""

def add(regs,p,ac):
    for i in range(32):
        if ac[i]:
            regs[i]+=p[1]
    return regs

def mul(regs,p,ac):
    for i in range(32):
        if ac[i]:
            regs[i]*=p[1]
    return regs

def if_it(regs,p,cycles,ac):
    ac1=[0]*32
    ac2=[0]*32
    flag1=0
    flag2=0
    for i in range(32):
        if(regs[i]<p[1] and ac[i]):
            ac1[i]=1
            flag1=1
        elif(regs[i]>=p[1] and ac[i]):
            ac2[i]=1
            flag2=1
    if(flag1):
        for p1 in p[2]:
            if p1[0]=="add":
                regs=add(regs,p1,ac1)
                cycles+=1
            elif p1[0]=="mul":
                regs=mul(regs,p1,ac1)
                cycles+=1
            else:
                regs,cycles=if_it(regs,p1,cycles,ac1)
    if(flag2):
        for p1 in p[3]:
            if p1[0]=="add":
                regs=add(regs,p1,ac2)
                cycles+=1
            elif p1[0]=="mul":
                regs=mul(regs,p1,ac2)
                cycles+=1
            else:
                regs,cycles=if_it(regs,p1,cycles,ac2)
    return (regs,cycles)

def run(program):
    cycles=0
    regs=list(range(32))
    ac=[1]*32
    for p in program:
        if p[0]=="add":
            regs=add(regs,p,ac)
            cycles+=1
        elif p[0]=="mul":
            regs=mul(regs,p,ac)
            cycles+=1
        else:
            regs,cycles=if_it(regs,p,cycles,ac)
    return (regs,cycles)
