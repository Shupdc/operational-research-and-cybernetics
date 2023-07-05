tic
N = 2000;

% 构造2000维系数方阵Q、列向量C
Q = eye(N); % Q半正定，凸
for i=2:2:N
    Q(i,i) = 2;
end

C = ones(1,N);
for i=2:2:N
    C(1,i)=2;
end
% 定义目标函数
f = @(x) 0.5 * x' * Q * x + C * x; 
% 更多的维数，运算时间，矩阵，fminunc

% 设置初始点和其他参数
% x0 = -1.1*ones(N,1);  % 初始点
x0 = rand(N, 1)+19;
alpha = 1;  % 步长的初始值y
beta = 0.5;  % 步长衰减系数
epsilon = 1e-4;  % 精度
maxIter = 1000;  % 最大迭代次数


% 调用梯度下降法函数

[x, fval, iter] = matrixDG_1(f, x0, alpha, beta, epsilon, maxIter);


% 输出结果
% disp(['Optimal point x = [', num2str(x(1)), ', ', num2str(x(2)), ']']);
disp(['Optimal value f(x) = ', num2str(fval)]);
disp(['Number of iterations = ', num2str(iter)]);

toc
