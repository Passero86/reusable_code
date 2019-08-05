function [ out ] = myKernel(x,lambda)
% linear kernel
if lambda == 2015
    out = x * x';
% polynomial kernel 0, 2
else if lambda == 2016
    out = x * x';
    out = out .^ 2;
% polynomial kernel 0, 4
else if lambda == 2017
    out = x*x';
    out = out .^4;
% polynomial kernel 1, 2
else if lambda == 2018      
    out = 1+x*x';
    out = out .^2; 
% polynomial kernel 1, 4
else if lambda == 2019
    out = 1+x*x';
    out = out .^4; 
% unknown kernel
else if lambda==2020
    m=size(x,1);
    out=zeros(m);
    for i=1:m
        for j=i+1:m
           out(i,j)=x(i,:)*x(j,:)'/(norm(x(i,:))*norm(x(j,:))+eps);
           out(j,i)=out(i,j);
        end
    end
% gaussian kernel      
else
    tmp=EuDist2(x,[],0);
    delta=lambda*max(tmp(:));
    out = exp(-tmp./delta);
    end
    end
    end
    end
    end
end
     % scaling
     maximum=max(out(:));
     out = out ./maximum;
   
end
