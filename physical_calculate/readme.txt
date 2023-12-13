输入文件是POSCAR
strain.py 1 0.995
意思是在x方向上压缩0.5%

strain.py 1 2 0.995
意思是在x和y方向上同时压缩0.5%

C11 就是在x方向压缩和拉伸，然后求E_p 对压缩百分比的二次导数
C12 就是在x和y方向同时压缩和拉伸，然后求E_p 对压缩百分比的二次导数

得现有poscar_0.990这些
for j in 1 2
do      
    for k in 0.990 0.995 1.000 1.005 1.010
    do
        echo start $j $k
        cp -ra 0 ${j}_${k}
        cd ${j}_${k}
        strain.py ${j} ${k}
        \mv poscar_${k} POSCAR
        cd ..
    done
done

这是算c11 和 c22的

另外计算E_0和B_v
需要对晶格进行-1.0% 到 1.0%的缩放

算E_p
然后放两列数字，我这里文件名是summary
两列数字分别是体积和e_p

然后执行eosase.py

strain.py 1 2 3 0.995
这种就可以对正交的POSCAR 做缩放了

当然你直接对PWmat格式的lattice，整体乘一个数字也行
一般是  0.990, 0.995, 1.000, 1.010, 1.005
这么几个数字

总之要算的所有东西都是，把晶格矢量变一下，共五个点，然后拟合抛物线，y = ax^2 + bx + c
\partial y / \parital x = 2a