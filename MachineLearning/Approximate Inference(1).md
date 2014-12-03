<!-- mathjax config similar to math.stackexchange -->
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
	
  jax: ["input/TeX", "output/HTML-CSS"],
  tex2jax: {
    inlineMath: [ ['$', '$'] ],
    displayMath: [ ['$$', '$$']],
    processEscapes: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  messageStyle: "none",
  "HTML-CSS": { preferredFont: "TeX", availableFonts: ["STIX","TeX"] },
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>
<script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML" type="text/javascript"></script>

<style type="text/css"> body{ font-size:20px; line-height:24px;} </style>

#概述
概率模型的核心工作是在有观测样本$X$时计算隐变量的posterior distribution $p(Z|X)$，同时计算$Z$的期望作为$Z$的估计。在full bayesian framework中，我们可以引入参数的先验分布，将待估计的参数也加入隐变量空间中。但是在大多数情况下，由于$Z$的维度很高，常常导致后验概率无法求得。因此需要考虑近似的方法来处理高维问题。

#variational inference framework
##问题建模：Inference
在fully Bayesian Model中，所有参数都有prior distribution。

定义

$Z$包含所有隐变量和参数变量，$Z=\{z_1,\dots,z_N\}$。

$X$表示输入样本，$X=\{x_1,\dots,x_N\}$。

同时，我们定义probabilistic model来表示joint distribution$p(X,Z)$，表示生成输入样本的假设。我们的目标是找到参数的后验概率$p(Z|X)$，得到模型的参数估计；以及evidence$p(X)$，表示probabilistic model的符合程度。

分解log marginal probabililty
\begin{equation}
lnp(X)=L(q)+KL(q||p)
\end{equation}
其中
\begin{equation}
L(q)=\int{q(Z)ln{\frac{p(X,Z)}{q(Z)}}dZ}
\end{equation}

\begin{equation}
KL(q||p)=-\int{q(Z)ln{\frac{p(X|Z)}{q(Z)}}dZ}
\end{equation}

与EM算法不同的是，这里已经没有了$\theta$，所有的参数都引入了先验分布，并入了隐变量$Z$。

在EM算法中
-(E step)：令$q(Z)=p(Z|X)$，最小化KL距离。
-(M step)：极大化$L(q)$ with respect to $\theta$，提升下界。

在这里，我们认为$p(Z|X)$是难以计算的，在E step难以得到$q(Z)$，因此引入近似的方法求解$q(Z)$。

##Factorized distribution
对于隐变量$Z$，分为若干个独立的元素组$Z_{i}$，其中$i=1,\dots,M$，可得

\begin{equation}
\label{Factor1}
q(Z)=\prod_{i=1}^M{q_i(Z_i)}
\end{equation}

极大化 $L(q)$ with respect to $q_j$
将($\ref{Factor1}$)代入$L(q)$ ,并将$q_j$分离出来
$$
L(q)=\int{\prod_i q_i \times {lnp(X,Z)-\sum_ilnq_i}}
$$
$$
=\int{q_j \times \{\int{lnp(X,Z) \times \prod_q_i dZ_i\}dZ_j}} - \int{q_j ln q_jdZ_j} + const \notag
$$
$$
=\int{q_j ln \widetilde{p} (X,Z_j) dZ_j} - \int{q_j lnq_j dZ_j} + const
$$

其中，我们定义
$$ln\widetilde{p}(X,Z_j)=E_{i \neq j}[lnp(X,Z)]+const 
$$

于是有

$$
\begin{equation}
\label{Va1}
E_{i \neq j}[lnp(X,Z)] = \int{lnp(X,Z) \times \prod q_i dZ_i}
\end{equation}
$$

$L(q)$的分解式为一个negative KL divergence，当$q_j(Z_j)=\widetilde{p}(X,Z_j)$时，取得最大值。因此极大化$L(q)$ with respect to $q_j$有以下解

\begin{equation}
\label{Va2}
lnq_j^*(Z_j)=E_{i \neq j}[lnp(X,Z)] + const
\end{equation}


##例子：the univariate Gaussian
###问题的定义

采用高斯分布模型

输入样本$D=\{x_1,\dots,x_N\}$

目标：infer参数的后验分布$p(\mu,\tau|D)$

###似然分布

高斯模型的likelihood function为

$$
p(D|\mu,\tau)=\left(\frac{\tau}{2\pi}\right)^{N/2}\exp \\{ -\frac{\tau}{2}\sum_{n=1}{N}(x_n-\mu)^2 \\}
$$

引入参数$\mu$和$\tau$的共轭先验分布

$$
p(\mu | \tau)=N(\mu | \mu_0, (\lambda_0 \tau)^{-1})
$$

$$
p(\tau)=Gam(\tau | a_0,b_0)
$$

因此高斯模型的joint distribution为

$$
p(D, \mu, \tau)=p(D|\mu, \tau)p(\mu | \tau)p(\tau)
$$


###Factorize
分解后验分布$p(\mu,\tau|D)$

$$q(\mu,\tau) = q\_{\mu}(\mu)  q\_{\mu}(\tau)$$

根据式($\ref{Va2}$)，可得$q\_{\mu}$与$q\_{\tau}$的最优估计

$$
lnq\_{\mu}^*(\mu)=E_r[\ln p(D|\mu,\tau) + \ln p(\mu | \tau)] + const
$$

$$
=-\frac{E[\tau]}{2} \\{\lambda_0 (\mu - \mu_0)^2 + \sum \limits\_{n=1}^N (x_n-\mu)^2\\} + const
$$

通过观察，我们可以发现$q_{\mu}(\mu)$，有关于$x$的二次项，服从高斯分布$N(\mu | \mu\_N, \lambda\_{N}^{-1})$，其中

$$
\mu_N=\frac{\lambda_0 \mu_0 + N\bar{x}}{\lambda_0 + N}
$$

$$
\lambda_N = (\lambda_0 + N)E[\tau]
$$

类似的，我们可以找到$q\_{\tau}(\tau)$的最优解

$$
\ln q\_{\tau}^{*} (\tau) = E\_{\mu}[\ln p(D|\mu, \tau) + \ln p(\mu | \tau)] + \ln p(\tau) + const
$$

$$
=(a_0-1) \ln \tau -b_0 \tau + \frac{N}{2} \ln \tau
$$

$$
 - \frac{\tau}{2} E\_{\mu}[\sum \limits_{n=1}^{N} (x_n - \mu)^2 + \lambda\_0 (\mu - \mu_0)^2] + const
$$

通过观察，$q\_{\tau}^{*}(\tau)$服从Gamma分布，其中

$$
a_N = a_0 + \frac{N}{2}
$$

$$
b_N = b_0 + \frac{1}{2}E\_{\mu}[\sum \limits_{n=1}^{N} (x_n - \mu)^2 + \lambda\_0 (\mu - \mu_0)^2]
$$

###Iteration
最后，我们利用迭代的方法得到最优解。

- Init
	1. 设置先验分布参数$\mu_0=a_0=b_0=\lambda_0=0$，表示没有先验假设。
- Iteration until convergence
	1. 利用$E\_{\mu}[\mu]$计算$q\_{\tau}(\tau)$ 
	1. 利用$q_\tau({\tau})$计算$E[\tau]$
	1. 利用$E[\tau]$计算$q\_{\mu}(\mu)$
	1. 利用$q\_{\mu}(\mu)$计算$E\_{\mu}[\mu]$

在本例中，由于没有隐变量，我们没有把迭代过程写成EM算法的形式。只是简单的迭代方法。

#总结
1. 有没有条件？
	回顾引入，我们有输入样本$D$，有概率模型表示的joint distribution$p(D,Z)$。
	
	目标是估计参数的后验分布$p(Z|D)$。
	
	困难是$Z$关系复杂，无法积分得到evidence$p(D)$。
	
	最后，我们将$Z$的分布估计为$q(Z)$，优化$L(q)$，也就是优化evidence$\ln p(D)$。
	
	当$\ln p(D)$是凸函数时，可以得到最优解。
	
	“***图模型变分推理是一种重要的确定性近似推理方法,根据凸对偶原理把概率推理问题转化为关于自由分布的泛函优化问题,并通过求解该优化问题进行近似推理。***”--<基于消息传播的图模型近似变分推理>
	
	这里讲的凸对偶原理就是对$p(D)$的分解。"转化为关于自由分布的泛函优化问题" 优化$L(q)$。
	
2. 标准流程
	
	变分推断的具体是怎么操作的，总结如下：
	
	a. 构建参数$Z$与样本$x_n$的生成关系$p(X|Z)$
	b. 为参数引入先验分布，通常与$p(X|Z)$构成共轭分布。以上两步可以表示为一个图模型。
	c. 引入基于参数分解的近似分布$q(Z)$，根据式($\ref{Va2}$)求解每一个分布$q_{j}(Z_j)$。在这里，由于我们引入的是先验共轭分布，因此后验分布为参数分布。




















