function [x, fval] = matrixDG_2(f, x0, alpha, beta, epsilon, maxIter)

    % f: 目标函数(矩阵形式）
    % x0: 初始点
    % alpha: 步长的初始值
    % beta: 步长衰减系数
    % epsilon: 精度
    % maxIter: 最大迭代次数
    x = x0;  
    fval = f(x); 
    gradNorm = norm(numericalGradient(x));  % 计算初始点的梯度范数
    iter = 0;     
    
    while gradNorm > epsilon && iter < maxIter
        iter = iter + 1;
        d = -numericalGradient(x);
        
        % Wolfe条件
        t = wolfeLineSearch(f, x, d, alpha, beta);
        
        x = x + t*d;
%         disp(x);
        fval = f(x);
        gradNorm = norm(numericalGradient(x));
    end
end


function t = wolfeLineSearch(f, x, d, alpha, beta)
    c1 = 1e-3;  % Armijo条件的常数
    c2 = 0.8;  % 曲率条件的常数
    t = alpha;  % 初始步长
    while true
        fval = f(x + t*d);
        gradDotDir = numericalGradient(x + t*d)' * d; % 梯度向量grad(x)与搜索方向d的内积
        fvalT = f(x) + c1 * t * numericalGradient(x)' * d; % 需要比较的值       
        if fval > fvalT % 如果没有充分下降？
            t = zoomLineSearch(f, x, d, t, c1, c2);
            break;
        end
        if abs(gradDotDir) <= c2 * abs(numericalGradient(x)' * d) % 是否满足强曲率条件？满足则退
            break;
        end
        t = beta * t;
    end
end


function grad = numericalGradient(x)
    global Q C
    grad = Q * x + C';
end

function t = zoomLineSearch(f, x, d, t0, c1, c2)
    % Wolfe条件的线搜索（zoom） 
    tLow = 0;
    tHigh = t0;
    f0 = f(x);
    fvalLow = f0; 
    t = 1;
    while true
        t = (tLow + tHigh) / 2;
        xNew = x + t*d;
        fval = f(xNew);
        
        if fval > f0 + c1 * t * numericalGradient(x)' * d || fval >= fvalLow  % 如果仍然不能充分下降，说明步长还不够小
            tHigh = t;
            
        else  % 如果满足了充分下降，是否过小？
            gradDotDir = numericalGradient(xNew)' * d;
            
            if abs(gradDotDir) <= c2 * abs(numericalGradient(x)' * d)
                break;
            end
            tLow = t;  % 更新下界
            fvalLow = fval;  % 更新较低函数值
        end
        if tLow == tHigh % 防止死循环
            break
        end
    end
end