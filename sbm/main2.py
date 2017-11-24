import func
for m in range(300,1000,100):
    reload(func)
    r = 2 
    A, labels = func.create_sbm(n=m,cin=10,cout=1,q=r)
    constraint = func.create_constraint(labels, 5*r)
    print(constraint)
    labels1 = func.unnorm_spec_clustering(A,r)
    labels2 = func.bethe_hessian_clustering(A,r)
    idx = func.fastge2 (A, r, constraint)
    print(func.nmi(labels,labels1))
    print(func.nmi(labels,labels2))
    print('')
