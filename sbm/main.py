import func
for m in range(1000,5000,100):
    reload(func)
    r = 30
    A, labels = func.create_sbm(n=m,cin=200,cout=10,q=r)
    labels1 = func.unnorm_spec_clustering(A,r)
    labels2 = func.bethe_hessian_clustering(A,r)
    print(func.nmi(labels,labels1))
    print(func.nmi(labels,labels2))
    print('')
