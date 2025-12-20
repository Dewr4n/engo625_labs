nn=zeros(1,2);
nn(1,:)=6;  % 7 * 7  loop
istr=sym('i',[1,length(nn)]);
str1='';
str2=['func(',char(istr),')'];
for i=1:length(nn)
    str1=[str1,'for ',char(istr(i)),'=-',num2str(nn(i)/2),':',num2str(nn(i)/2),','];
    str2=[str2,';end'];
end
str1
str2
strFor=[str1,str2]
tic
eval(strFor)
timeElapsed = toc
function func(I)
    disp(I)
end