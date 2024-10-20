# Aim: To Implement Linear Regression

from math import sqrt
import seaborn as sns    #pip install seaborn
import matplotlib.pyplot as plt     #pip install matplotlib

x=list(map(int, input ("Enter X data:").split(",")))
y=list(map(int, input ("Enter Y data:").split(",")))

n=len(x)

xmean=sum(x)/n
ymean=sum(y)/n

a=[]
b=[]
for i in range(n):
    a.append(x[i]-xmean)
    b.append(y[i]-xmean)

ab=[a[i]*b[i]for i in range(n)]
asqaure=[a[i]**2 for i in range (n)]

bsquare = [b[i]**2 for i in range (n)]
r = sum (ab)/sqrt(sum(asqaure)*sum(asqaure))

dely=sqrt(sum(bsquare))/sqrt(n-1)
delx=sqrt(sum(bsquare))/sqrt(n-1)

b1=r*dely/delx
b0=ymean-b1*xmean

print("B0:",b0,"B1",b1)
print("Equation: y=",b0,"+",b1,"x")

sns.scatterplot(x=x,y=y)
xpred=[i for i in range (min(x),max(x)+1)]
ypred=[b0+(b1*i)for i in xpred]
sns.lineplot(x=xpred,y=ypred)

plt.show()