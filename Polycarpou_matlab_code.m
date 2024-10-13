
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NAME OF FILE: FEM1DL.m
%
% PURPOSE: This Matlab code solves a one-dimensional boundary-value problem
% using the finite element method. The boundary-value problem at hand is
% defined by the Poisson's equation in a simple medium (isotropic, linear,
% and homogeneous) characterized by a uniform electron charge density rhov=-rho0. 
% The finite element method is based on sub-dividing the one-dimensional domain 
% into Ne linear elements of equal length. Two Dirichlet boundary conditions are 
% applied: V=Va at the leftmost node of the domain, and V=Vb at the rightmost 
% node of the domain. The length of the domain is denoted by L. The user has 
% the freedom to modify any of the defined input variables including charge 
% density (rho0), domain length (L), boundary conditions (Va & Vb), dielectric 
% constant (epsr), and number of elements (Ne).
%
% All quantities are expressed in the SI system of units.
%
% The Primary Unknown Quantity is the Electric Potential.
% The Secondary Unknown Quantity is the Electric Field.
%
% Written by Anastasis Polycarpou (Last updated: 8/12/2005)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear % Clear all variables
%
% Define the Input Variables
% ==========================
L=8*10^-2; % Length of the domain
rho0=1*10^-8; % Charge density
epsr=1.0; % Dielectric constant of the domain
Va=1; % Boundary condition at the leftmost node
Vb=0; % Boundary condition at the rightmost node
Ne=4; % Number of linear elements
% ==========================
eps=epsr*8.85*10^-12;
Nn=Ne+1;
for i=1:Ne
    elmconn(i,1)=i;
    elmconn(i,2)=i+1;
end
le=L/Ne;
Ke(1,1)=eps/le;
Ke(1,2)=-eps/le;
Ke(2,1)=-eps/le;
Ke(2,2)=eps/le;
fe=-le*rho0/2*([1;1]);
K=zeros(Nn);
f=zeros(Nn,1);
for e=1:Ne
    for i=1:2
        for j=1:2
            K(elmconn(e,i),elmconn(e,j))=K(elmconn(e,i),elmconn(e,j))+Ke(i,j);
        end
        f(elmconn(e,i))=f(elmconn(e,i))+fe(i);
    end
end
for i=2:Nn
    f(i)=f(i)-K(i,1)*Va;
end
K(:,1)=0;
K(1,:)=0;
K(1,1)=1;
f(1)=Va;

for i=2:Nn-1
    f(i)=f(i)-K(i,Nn)*Vb;
end
K(:,Nn)=0;
K(Nn,:)=0;
K(Nn,Nn)=1;
f(Nn)=Vb;

V=K\f;

for e=1:Ne
    x(e,1)=(e-1)*le;
    x(e,2)=e*le;
end

Npoints=50;
dx=le/(Npoints);
for e=1:Ne
    for i=1:Npoints
        idx=(e-1)*Npoints+i;
        xeval(idx)=(idx-1)*dx;
        xeval2(idx+e-1)=(idx-1)*dx;
        Veval(idx)=V(e)*(x(e,2)-xeval(idx))/le+V(e+1)*(xeval(idx)-x(e,1))/le;
        Eeval(idx+e-1)=(V(e)-V(e+1))/le;
    end
    if e < Ne
        xeval2(idx+e)=xeval2(idx+e-1);
        Eeval(idx+e)=(V(e+1)-V(e+2))/le;
    end
end
xeval(idx+1)=idx*dx;
xeval2(idx+e)=idx*dx;
Veval(idx+1)=V(Ne+1);
Eeval(idx+e)=Eeval(idx+e-1);
plot(xeval,Veval,'k--'); % Plot the Electric potential obtained from the finite element method
%
% Exact Analytical Solution
% =========================
hold
for i=1:Ne*Npoints+1
    Vexact(i)=rho0/(2*eps)*xeval(i)^2+((Vb-Va)/L-rho0*L/(2*eps))*xeval(i)+Va;
    Eexact(i)=rho0*(L-2*xeval(i))/(2*eps)+(Va-Vb)/L;
end
plot(xeval,Vexact,'k-'); % Plot the Electric potential obtained from the exact analytical solution
xlabel('x (meters)');
ylabel('V (Volts)');
legend('FEM','Exact');
%
% Error Analysis (Using the area bounded between the two curves per element)
% =========================================================================
PercentError=0.0;
Aextot=0.0;
Diff=0.0;
for e=1:Ne
    Aexact=rho0/(6*eps)*(x(e,2)^3-x(e,1)^3)-0.5*(rho0*L/(2*eps)+Va/L)*(x(e,2)^2-x(e,1)^2)+Va*(x(e,2)-x(e,1));
    Aextot=Aextot+Aexact;
    Afe=le/2*(V(e)+V(e+1));
    Diff=Diff+abs(Aexact-Afe);
end
Aextot=abs(Aextot);
PercentError=Diff/Aextot*100
%
% Error Analysis (Using the L2-Norm Definition)
% ============================================
L2Error=0.0;
for e=1:Ne
    f1=rho0^2*(x(e,2)^5-x(e,1)^5)/(20*eps^2);
    f2=(V(e)/le-rho0*L/(2*eps)-Va/L-V(e+1)/le)*rho0/(4*eps)*(x(e,2)^4-x(e,1)^4);
    f3=(1/3)*((V(e+1)*x(e,1)/(eps*le)+Va/eps-V(e)*x(e,2)/(eps*le))*rho0+(V(e)/le-rho0*L/(2*eps)-Va/L-V(e+1)/le)^2)*(x(e,2)^3-x(e,1)^3);
    f4=(V(e+1)*x(e,1)/le+Va-V(e)*x(e,2)/le)*(V(e)/le-rho0*L/(2*eps)-Va/L-V(e+1)/le)*(x(e,2)^2-x(e,1)^2);
    f5=(V(e+1)*x(e,1)/le+Va-V(e)*x(e,2)/le)^2*(x(e,2)-x(e,1));
    L2Error=L2Error+(f1+f2+f3+f4+f5);
end
L2Error=sqrt(L2Error)


figure(2);
plot(xeval2,Eeval,'k--'); % Plot the Electric field obtained from the finite element method
hold
plot(xeval,Eexact,'k-'); % Plot the Electric field obtained from the exact analytical solution
xlabel('x (meters)');
ylabel('E (V/m)');
legend('FEM','Exact');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NAME OF FILE: FEM1DQ.m
%
% PURPOSE: This Matlab code solves a one-dimensional boundary-value problem
% using the finite element method. The boundary-value problem at hand is
% defined by the Poisson's equation in a simple medium (isotropic, linear,
% and homogeneous) characterized by a uniform electron charge density rhov=-rho0. 
% The finite element method is based on sub-dividing the one-dimensional domain 
% into Ne quadratic elements of equal length. Two Dirichlet boundary conditions 
% are applied: V=Va at the leftmost node of the domain, and V=Vb at the rightmost 
% node of the domain. The length of the domain is denoted by L. The user has 
% the freedom to modify any of the defined input variables including charge 
% density (rho0), domain length (L), boundary conditions (Va & Vb), dielectric 
% constant (epsr), and number of elements (Ne).
%
% All quantities are expressed in the SI system of units.
%
% The Primary Unknown Quantity is the Electric Potential.
% The Secondary Unknown Quantity is the Electric Field.
%
% Written by Anastasis Polycarpou (Last updated: 8/12/2005)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear % Clear all variables
%
% Define the Input Variables
% ==========================
L=8*10^-2; % Length of the domain
rho0=1*10^-8; % Charge density
epsr=1.0; % Dielectric constant of the domain
Va=1; % Boundary condition at the leftmost node
Vb=0; % Boundary condition at the rightmost node
Ne=4; % Number of quadratic elements
% ==========================
eps=epsr*8.85*10^-12;
Nn=2*Ne+1;
for e=1:Ne
    elmconn(e,1)=2*e-1;
    elmconn(e,2)=2*e+1;
    elmconn(e,3)=2*e;
end
le=L/Ne;
Ke(1,1)=7*eps/(3*le);
Ke(1,2)=eps/(3*le);
Ke(1,3)=-8*eps/(3*le);
Ke(2,1)=eps/(3*le);
Ke(2,2)=7*eps/(3*le);
Ke(2,3)=-8*eps/(3*le);
Ke(3,1)=-8*eps/(3*le);
Ke(3,2)=-8*eps/(3*le);
Ke(3,3)=16*eps/(3*le);
fe=-le*rho0/6*([1;1;4]);
K=zeros(Nn);
f=zeros(Nn,1);
for e=1:Ne
    for i=1:3
        for j=1:3
            K(elmconn(e,i),elmconn(e,j))=K(elmconn(e,i),elmconn(e,j))+Ke(i,j);
        end
        f(elmconn(e,i))=f(elmconn(e,i))+fe(i);
    end
end
for i=2:Nn
    f(i)=f(i)-K(i,1)*Va;
end
K(:,1)=0;
K(1,:)=0;
K(1,1)=1;
f(1)=Va;

for i=2:Nn-1
    f(i)=f(i)-K(i,Nn)*Vb;
end
K(:,Nn)=0;
K(Nn,:)=0;
K(Nn,Nn)=1;
f(Nn)=Vb;

V=K\f;
for e=1:Ne
    x(e,1)=(2*e-2)*le/2;
    x(e,2)=(2*e)*le/2;
    x(e,3)=(2*e-1)*le/2;
end

Npoints=10;
dx=le/(Npoints);
for e=1:Ne
    for i=1:Npoints
        idx=(e-1)*Npoints+i;
        xeval(idx)=(idx-1)*dx;
        ksi=2*(xeval(idx)-x(e,3))/le;
        Veval(idx)=V(2*e-1)*0.5*ksi*(ksi-1)+V(2*e+1)*0.5*ksi*(ksi+1)+V(2*e)*(1+ksi)*(1-ksi);
        Eeval(idx)=(-2/le)*(V(2*e-1)*(ksi-0.5)+V(2*e+1)*(ksi+0.5)+V(2*e)*(-2*ksi));
    end
end
xeval(idx+1)=idx*dx;
Veval(idx+1)=V(2*Ne+1);
ksi=2*(xeval(idx+1)-x(Ne,3))/le;
Eeval(idx+1)=(-2/le)*(V(2*Ne-1)*(ksi-0.5)+V(2*Ne+1)*(ksi+0.5)+V(2*Ne)*(-2*ksi));
plot(xeval,Veval,'k--'); % Plot the Electric potential obtained from the finite element method
%
% Exact Analytical Solution
% =========================
hold
for i=1:Ne*Npoints+1
    Vexact(i)=rho0/(2*eps)*xeval(i)^2+((Vb-Va)/L-rho0*L/(2*eps))*xeval(i)+Va;
    Eexact(i)=rho0*(L-2*xeval(i))/(2*eps)+(Va-Vb)/L;
end
plot(xeval,Vexact,'k-'); % Plot the Electric potential obtained from the exact analytical solution
xlabel('x (meters)');
ylabel('V (volts)');
legend('FEM','Exact');
%
% Error Analysis (Using the area bounded between the two curves per element)
% =========================================================================
PercentError=0.0;
Aextot=0.0;
Diff=0.0;
for e=1:Ne
    g1=rho0/(6*eps)*(x(e,3)^3-x(e,1)^3);
    g2=0.5*(rho0*L/(2*eps)+Va/L)*(x(e,3)^2-x(e,1)^2);
    g3=Va*(x(e,3)-x(e,1));
    A1exact=g1-g2+g3;
    A1fe=le/24*(5*V(2*e-1)-V(2*e+1)+8*V(2*e));
    Diff=Diff+abs(A1exact-A1fe);
    g1=rho0/(6*eps)*(x(e,2)^3-x(e,3)^3);
    g2=0.5*(rho0*L/(2*eps)+Va/L)*(x(e,2)^2-x(e,3)^2);
    g3=Va*(x(e,2)-x(e,3));
    A2exact=g1-g2+g3;
    A2fe=le/24*(-V(2*e-1)+5*V(2*e+1)+8*V(2*e));
    Diff=Diff+abs(A2exact-A2fe);
    Aextot=Aextot+A1exact+A2exact;
end
Aextot=abs(Aextot);
PercentError=Diff/Aextot*100

figure(2);
plot(xeval,Eeval,'k--'); % Plot the Electric field obtained from the finite element method
hold
plot(xeval,Eexact,'k-'); % Plot the Electric field obtained from the exact analytical solution
xlabel('x (meters)');
ylabel('E (V/m)');
legend('FEM','Exact');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NAME OF FILE: FEM1DQnucd.m
%
% PURPOSE: This Matlab code solves a one-dimensional boundary-value problem
% using the finite element method. The boundary-value problem at hand is
% defined by the Poisson's equation in a simple medium (isotropic, linear,
% and homogeneous) characterized by a non-uniform electron charge density 
% described by 
%                            rho=-rho0*(1-x/L)^2 
% The finite element method is based on sub-dividing the one-dimensional 
% domain into Ne quadratic elements of equal length. Two Dirichlet boundary 
% conditions are applied: V=Va at the leftmost node of the domain, and V=Vb at 
% the rightmost node of the domain. The length of the domain is denoted by L. 
% The user has the freedom to modify any of the defined input variables 
% including charge density (rho0), domain length (L), boundary conditions 
% (Va & Vb), dielectric constant (epsr), and number of elements (Ne).
%
% All quantities are expressed in the SI system of units.
%
% The Primary Unknown Quantity is the Electric Potential.
% The Secondary Unknown Quantity is the Electric Field.
%
% Written by Anastasis Polycarpou (Last updated: 8/12/2005)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear % Clear all variables
%
% Define the Input Variables
% ==========================
L=8*10^-2; % Length of the domain
rho0=1*10^-8; % Charge density
epsr=1.0; % Dielectric constant of the domain
Va=1; % Boundary condition at the leftmost node
Vb=0; % Boundary condition at the rightmost node
Ne=4; % Number of elements
% ==========================
eps=epsr*8.85*10^-12;
Nn=2*Ne+1;
le=L/Ne;
for e=1:Ne
    elmconn(e,1)=2*e-1;
    elmconn(e,2)=2*e+1;
    elmconn(e,3)=2*e;
    x(e,1)=(2*e-2)*le/2;
    x(e,2)=(2*e)*le/2;
    x(e,3)=(2*e-1)*le/2;
end
Ke(1,1)=7*eps/(3*le);
Ke(1,2)=eps/(3*le);
Ke(1,3)=-8*eps/(3*le);
Ke(2,1)=eps/(3*le);
Ke(2,2)=7*eps/(3*le);
Ke(2,3)=-8*eps/(3*le);
Ke(3,1)=-8*eps/(3*le);
Ke(3,2)=-8*eps/(3*le);
Ke(3,3)=16*eps/(3*le);
K=zeros(Nn);
f=zeros(Nn,1);
for e=1:Ne
    alpha=1-x(e,3)/L;
    f1=-le*rho0/4*(le^2/(10*L^2)+2*alpha*le/(3*L)+(2/3)*alpha^2);
    f2=-le*rho0/4*(le^2/(10*L^2)-2*alpha*le/(3*L)+(2/3)*alpha^2);
    f3=-le*rho0/2*(le^2/(15*L^2)+(4/3)*alpha^2);
    fe=([f1;f2;f3]);
    for i=1:3
        for j=1:3
            K(elmconn(e,i),elmconn(e,j))=K(elmconn(e,i),elmconn(e,j))+Ke(i,j);
        end
        f(elmconn(e,i))=f(elmconn(e,i))+fe(i);
    end
end
for i=2:Nn
    f(i)=f(i)-K(i,1)*Va;
end
K(:,1)=0;
K(1,:)=0;
K(1,1)=1;
f(1)=Va;

for i=2:Nn-1
    f(i)=f(i)-K(i,Nn)*Vb;
end
K(:,Nn)=0;
K(Nn,:)=0;
K(Nn,Nn)=1;
f(Nn)=Vb;

V=K\f;

Npoints=10;
dx=le/(Npoints);
for e=1:Ne
    for i=1:Npoints+1
        idx=(e-1)*(Npoints+1)+i;
        xeval(idx)=(idx-e)*dx;
        ksi=2*(xeval(idx)-x(e,3))/le;
        Veval(idx)=V(2*e-1)*0.5*ksi*(ksi-1)+V(2*e+1)*0.5*ksi*(ksi+1)+V(2*e)*(1+ksi)*(1-ksi);
        Eeval(idx)=(-2/le)*(V(2*e-1)*(ksi-0.5)+V(2*e+1)*(ksi+0.5)+V(2*e)*(-2*ksi));
    end
end
plot(xeval,Veval,'k--'); % Plot the Electric potential obtained from the finite element method 
%
% Exact Analytical Solution
% =========================
hold
for i=1:Ne*(Npoints+1) 
    Vexact(i)=L^2*rho0/(12*eps)*(1-xeval(i)/L)^4+(L*rho0/(12*eps)+(Vb-Va)/L)*xeval(i)+(Va-L^2*rho0/(12*eps));
    Eexact(i)=L*rho0/(3*eps)*(1-xeval(i)/L)^3-(L*rho0/(12*eps)+(Vb-Va)/L);
end
plot(xeval,Vexact,'k-'); % Plot the Electric potential obtained from the exact analytical solution
xlabel('x (meters)');
ylabel('V (volts)');
legend('FEM', 'Exact');
%
% Error Analysis (Using the area bounded between the two curves per element)
% =========================================================================
PercentError=0.0;
Aextot=0.0;
Diff=0.0;
for e=1:Ne
    g1=rho0*(x(e,3)^5-x(e,1)^5)/(60*L^2*eps);
    g2=rho0*(x(e,3)^4-x(e,1)^4)/(12*L*eps);
    g3=rho0*(x(e,3)^3-x(e,1)^3)/(6*eps);
    g4=0.5*(rho0*L/(4*eps)+Va/L)*(x(e,3)^2-x(e,1)^2);
    g5=Va*(x(e,3)-x(e,1));
    A1exact=g1-g2+g3-g4+g5;
    A1fe=le/24*(5*V(2*e-1)-V(2*e+1)+8*V(2*e));
    Diff=Diff+abs(A1exact-A1fe);
    g1=rho0*(x(e,2)^5-x(e,3)^5)/(60*L^2*eps);
    g2=rho0*(x(e,2)^4-x(e,3)^4)/(12*L*eps);
    g3=rho0*(x(e,2)^3-x(e,3)^3)/(6*eps);
    g4=0.5*(rho0*L/(4*eps)+Va/L)*(x(e,2)^2-x(e,3)^2);
    g5=Va*(x(e,2)-x(e,3));
    A2exact=g1-g2+g3-g4+g5;
    A2fe=le/24*(-V(2*e-1)+5*V(2*e+1)+8*V(2*e));
    Diff=Diff+abs(A2exact-A2fe);
    Aextot=Aextot+A1exact+A2exact;
end
Aextot=abs(Aextot);
PercentError=Diff/Aextot*100


figure(2);
plot(xeval,Eeval,'k--'); % Plot the Electric field obtained from the finite element method
hold
plot(xeval,Eexact,'k-'); % Plot the Electric field obtained from the exact analytical solution
xlabel('x (meters)');
ylabel('E (V/m)');
legend('FEM','Exact');






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NAME OF FILE: FEM2DL_Box.m
%
% PURPOSE: This Matlab code solves a two-dimensional boundary-value problem
% using the finite element method. The boundary-value problem at hand
% corresponds to the Laplace's equation as applied to a rectangular domain 
% characterized by a simple medium (isotropic, linear, and homogeneous)
% with dielectric constant epsr. The finite element method is based on sub-
% dividing the two-dimensional domain into Ne linear triangular elements. 
% Dirichlet boundary conditions are applied to all four metallic walls: 
%
% V=0 on the left sidewall 
% V=0 on the right sidewall
% V=0 on the bottom sidewall
% V=V0 on the top wall which is separated by tiny gaps from the two sidewalls
%
% The dimensions of the domain are WxH, where W=Width and H=Hight. 
% The user has the freedom to modify any of the defined input variables 
% including domain width and height (W & L), top wall voltage (V0), and number 
% of bricks along the x- and y-axes. Note that each brick is then subdivided 
% into two triangles.
%
% All quantities are expressed in the SI system of units.
%
% The Primary Unknown Quantity is the Electric Potential.
% The Secondary Unknown Quantity is the Electric Field.
%
% Written by Anastasis Polycarpou (Last updated: 8/12/2005)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear % Clear all variables

% Define Rectangular Geometry
% ===========================
W=1;
H=1;

% Define the number of bricks in the x and y directions
% =====================================================
XBRICKS=20;
YBRICKS=20;

% Define the Voltage (Electric potential) on the top wall
% =======================================================
V0=1;

% Generate the triangular mesh
% ============================
XLLC=0;  % x-coordinate of the Lower Left Corner
YLLC=0;  % y-coordinate of the Lower Left Corner
XURC=W;  % x-coordinate of the Upper Right Corner
YURC=H;  % y-coordinate of the Upper Right Corner
TBRICKS=XBRICKS*YBRICKS;   % Total number of bricks
Dx=(XURC-XLLC)/XBRICKS;  % Edge length along x-direction
Dy=(YURC-YLLC)/YBRICKS;  % Edge length along y-direction

TNNDS=(XBRICKS+1)*(YBRICKS+1);    % Total number of nodes
TNELMS=2*TBRICKS;    % Total number of triangular elements
%
% Print the number of nodes and the number of elements
% ====================================================
fprintf('The number of nodes is: %6i\n',TNNDS)
fprintf('The number of elements is: %6i\n',TNELMS)

for I=1:TNNDS
    X(I)=mod(I-1,XBRICKS+1)*Dx;
    Y(I)=floor((I-1)/(XBRICKS+1))*Dy;
end

% Connectivity Information
% ========================
for I=1:TBRICKS
    ELMNOD(2*I-1,1)=I+floor((I-1)/XBRICKS);
    ELMNOD(2*I-1,2)=ELMNOD(2*I-1,1)+1+(XBRICKS+1);
    ELMNOD(2*I-1,3)=ELMNOD(2*I-1,1)+1+(XBRICKS+1)-1;
    
    ELMNOD(2*I,1)=I+floor((I-1)/XBRICKS);
    ELMNOD(2*I,2)=ELMNOD(2*I,1)+1;
    ELMNOD(2*I,3)=ELMNOD(2*I,1)+1+(XBRICKS+1);
end
%
% Plot the mesh
% =============
figure(1)
triplot(ELMNOD,X,Y);

% Define the constants for each element
% =====================================
for e=1:TNELMS
    alphax(e)=1;
    alphay(e)=1;
    beta(e)=0;
    g(e)=0;
end

% Definition of Dirichlet Boundary Conditions
% ===========================================
TNEBC=0;
for I=1:TNNDS
    if X(I) == XLLC || X(I) == XURC || Y(I) == YLLC
        TNEBC=TNEBC+1;
        EBCNOD(TNEBC)=I;
        EBCVAL(TNEBC)=0;
    elseif Y(I) == YURC
        TNEBC=TNEBC+1;
        EBCNOD(TNEBC)=I;
        EBCVAL(TNEBC)=V0;
    end
end

% Definition of Mixed Boundary Conditions
% =======================================
TNMBC=0;

% Initialization of the global K matrix and right-hand side vector
% ================================================================
K=sparse(TNNDS,TNNDS);
b=zeros(TNNDS,1);

% Form the element matrices and assemble to the global matrix
% ===========================================================
for e=1:TNELMS
    x21=X(ELMNOD(e,2))-X(ELMNOD(e,1));
    x31=X(ELMNOD(e,3))-X(ELMNOD(e,1));
    x32=X(ELMNOD(e,3))-X(ELMNOD(e,2));
    x13=X(ELMNOD(e,1))-X(ELMNOD(e,3));
    y12=Y(ELMNOD(e,1))-Y(ELMNOD(e,2));
    y21=Y(ELMNOD(e,2))-Y(ELMNOD(e,1));
    y31=Y(ELMNOD(e,3))-Y(ELMNOD(e,1));
    y23=Y(ELMNOD(e,2))-Y(ELMNOD(e,3));
    Ae=0.5*(x21*y31-x31*y21);
    
    % Evaluation of the element K matrix
    % ==================================
    Me(1,1)=-(alphax(e)*y23^2+alphay(e)*x32^2)/(4*Ae);
    Me(1,2)=-(alphax(e)*y23*y31+alphay(e)*x32*x13)/(4*Ae);
    Me(2,1)=Me(1,2);
    Me(1,3)=-(alphax(e)*y23*y12+alphay(e)*x32*x21)/(4*Ae);
    Me(3,1)=Me(1,3);
    Me(2,2)=-(alphax(e)*y31^2+alphay(e)*x13^2)/(4*Ae);
    Me(2,3)=-(alphax(e)*y31*y12+alphay(e)*x13*x21)/(4*Ae);
    Me(3,2)=Me(2,3);
    Me(3,3)=-(alphax(e)*y12^2+alphay(e)*x21^2)/(4*Ae);
    
    % Evaluation of the element T matrix
    % ==================================
    Te(1,1)=beta(e)*Ae/6;
    Te(1,2)=beta(e)*Ae/12;
    Te(2,1)=beta(e)*Ae/12;
    Te(1,3)=beta(e)*Ae/12;
    Te(3,1)=beta(e)*Ae/12;
    Te(2,2)=beta(e)*Ae/6;
    Te(2,3)=beta(e)*Ae/12;
    Te(3,2)=beta(e)*Ae/12;
    Te(3,3)=beta(e)*Ae/6;
    
    % Sum the element matrices Me and Te
    % ==================================
    for I=1:3
        for J=1:3
            Ke(I,J)=Me(I,J)+Te(I,J);
        end
    end
    
    % Evaluation of element vector ge
    % ===============================
    ge(1)=g(e)*Ae/3;
    ge(2)=g(e)*Ae/3;
    ge(3)=g(e)*Ae/3;
    
    % Evaluation of the element vector pe & update of the element K-matrix
    % ====================================================================
        % There is no boundary Gamma2 !!!!
    
    % Assemble element matrices & vectors into the global K matrix and b
    % vector
    % ==================================================================
    for I=1:3
        for J=1:3
            K(ELMNOD(e,I),ELMNOD(e,J))=K(ELMNOD(e,I),ELMNOD(e,J))+Ke(I,J);
        end
        b(ELMNOD(e,I))=b(ELMNOD(e,I))+ge(I);
    end
end

% Imposition of Dirichlet boundary conditions
% ===========================================
for I=1:TNEBC
    for J=1:TNNDS
        if J ~= EBCNOD(I) 
            b(J)=b(J)-K(J,EBCNOD(I))*EBCVAL(I);
        end
    end
    K(:,EBCNOD(I))=0;
    K(EBCNOD(I),:)=0;
    K(EBCNOD(I),EBCNOD(I))=1;
    b(EBCNOD(I))=EBCVAL(I);
end

% Solution of the global matrix system
% ====================================
V=K\b;

% Generate the solution over a grid and plot it
% =============================================
[Xgrid,Ygrid]=meshgrid(XLLC:0.01*(XURC-XLLC):XURC,YLLC:0.01*(YURC-YLLC):YURC);
Vgrid=zeros(101,101);
for I=1:101
    for J=1:101
        for e=1:TNELMS
         
            x2p=X(ELMNOD(e,2))-Xgrid(I,J);
            x3p=X(ELMNOD(e,3))-Xgrid(I,J);
            y2p=Y(ELMNOD(e,2))-Ygrid(I,J);
            y3p=Y(ELMNOD(e,3))-Ygrid(I,J);
            A1=0.5*abs(x2p*y3p-x3p*y2p);
                
            x2p=X(ELMNOD(e,2))-Xgrid(I,J);
            x1p=X(ELMNOD(e,1))-Xgrid(I,J);
            y2p=Y(ELMNOD(e,2))-Ygrid(I,J);
            y1p=Y(ELMNOD(e,1))-Ygrid(I,J);
            A2=0.5*abs(x2p*y1p-x1p*y2p);
                
            x1p=X(ELMNOD(e,1))-Xgrid(I,J);
            x3p=X(ELMNOD(e,3))-Xgrid(I,J);
            y1p=Y(ELMNOD(e,1))-Ygrid(I,J);
            y3p=Y(ELMNOD(e,3))-Ygrid(I,J);
            A3=0.5*abs(x1p*y3p-x3p*y1p);
                
            x21=X(ELMNOD(e,2))-X(ELMNOD(e,1));
            x31=X(ELMNOD(e,3))-X(ELMNOD(e,1));
            y21=Y(ELMNOD(e,2))-Y(ELMNOD(e,1));
            y31=Y(ELMNOD(e,3))-Y(ELMNOD(e,1));
            Ae=0.5*(x21*y31-x31*y21);
                
            if abs(Ae-(A1+A2+A3)) < 0.00001*Ae   
                 ksi=(y31*(Xgrid(I,J)-X(ELMNOD(e,1)))-x31*(Ygrid(I,J)-Y(ELMNOD(e,1))))/(2*Ae);
                 ita=(-y21*(Xgrid(I,J)-X(ELMNOD(e,1)))+x21*(Ygrid(I,J)-Y(ELMNOD(e,1))))/(2*Ae);
                 N1=1-ksi-ita;
                 N2=ksi;
                 N3=ita;
                 Vgrid(I,J)=N1*V(ELMNOD(e,1))+N2*V(ELMNOD(e,2))+N3*V(ELMNOD(e,3));
            end
        end
    end
end
 
% Plot the finite element solution of V using a contour plot
% ==========================================================
figure(2)
contourf(Xgrid,Ygrid,Vgrid,15);
xlabel('x');
ylabel('y');
colorbar;

% Exact Analytical Solution
% =========================
N=100;
Vexact=zeros(101,101);
for i=1:101
    for j=1:101
        for k=1:N
            Update=(4/pi)*sin((2*k-1)*pi/W*Xgrid(i,j))*sinh((2*k-1)*pi/W*Ygrid(i,j))/((2*k-1)*sinh((2*k-1)*pi*H/W));
            if abs(Update) > abs(0.001*Vexact(i,j))
                Vexact(i,j)=Vexact(i,j)+Update;
            end
        end
    end
end

% Error based on L2 Norm
% ======================
L2error=0;
for i=1:101
    for j=1:101
        L2error=L2error+(Vexact(i,j)-Vgrid(i,j))^2;
    end
end
L2error=sqrt(L2error/(101*101))






    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NAME OF FILE: FEM2DL_Cyl.m
%
% PURPOSE: This Matlab code solves a two-dimensional scattering problem
% using the finite element method. The scatterer is a perfectly
% conducting circular cylinder of radius robj; all the dimensions are given
% in terms of the free-space wavelength. A TM-to-z incident plane wave is 
% scattered from the circular cylinder and propagates undisturbed toward 
% infinity. To simulate this undisturbed propagation of the scattered field
% toward infinity, a first-order absorbing boundary condition (ABC) was
% imposed at a distance rho away from the center of the cylinder. The farther
% the ABC boundary is placed, the more accurate the ABC is. The free space 
% between the circular cylinder and the ABC boundary is subdivided into 
% triangles governed by linear interpolation functions. Dirichlet boundary 
% conditions are applied on the surface of the cylinder; i.e., the tangential 
% electric field, in this case Ez, is set to zero for all the nodes that 
% coincide with the surface of the circular cylinder.
%
% The finite element method is applied to the homogeneous scalar wave
% equation, otherwise known as the homogeneous Helmholtz equation. The
% primary unknown quantity is the total electric field in the z-direction
% which is given by the incident field plus the scattered field. The
% direction of the incident field is set toward the positive x-axis (phi_i=0) 
% whereas the total field is evaluated at a distance half-way between the 
% scatterer and the ABC boundary for all observation angles between 0 and 360 
% degrees. The numerical solution is compared with the exact analytical 
% solution.
%
% The user is allowed to set the following input parameters:
%
% rhoj = radius of the scatterer (circular cylinder) in wavelengths
% rho  = radius of the ABC boundary in wavelengths
% h    = discritization size in wavelengths
% E0   = amplitude of the incident electric field
%
% IMPORTANT: Depending on the number of nodes in the domain and the clock 
% speed of your computer, the finite element code may take several minutes 
% to execute. Try not to exceed 5,000 nodes otherwise you have to wait a 
% significant amount of time to get results!
%
% Written by Anastasis Polycarpou (Last updated: 8/12/2005)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear % Clear all variables
%
% Define the Input parameters 
% ===========================
robj=0.5; % Set the radius of the circular PEC cylinder in wavelengths
rho=1.5; % Set the radius of the outer circular ABC boundary in wavelengths
h=0.04; % Set the discretization size in wavelengths
E0=1; % Set the amplitude of the Incident field
% ===========================
%
% Determine the number of annular rings
% =====================================
Nseg=ceil((rho-robj)/(sqrt(3)*h/2));
dr=(rho-robj)/Nseg;
%
% Create the nodes on each annular ring
% =====================================
Nnods=0;
for J=1:Nseg+1
    r=robj+(J-1)*dr;
    Npoints=ceil(2*pi*r/h);
    dphi=2*pi/Npoints;
    for JJ=1:Npoints
        Nnods=Nnods+1;
        phi=(JJ-1)*dphi;
        x(Nnods)=r*cos(phi);
        y(Nnods)=r*sin(phi);
    end
end
%
% Triangulate using Delaunay triangulation
% ========================================
tri=delaunay(x,y);
%
% Eliminate the triangles within the perfectly conducting circular cylinder
% =========================================================================
Nelms=0;
for J=1:length(tri)
    Innods=0;
    for JJ=1:3
        r0=sqrt(x(tri(J,JJ))^2+y(tri(J,JJ))^2);
        if abs(robj-r0) < 0.0001*robj
            Innods=Innods+1;
        end
    end
    if Innods <= 2
        Nelms=Nelms+1;
        elmnod(Nelms,:)=tri(J,:);
    end
end
%
% Print the number of nodes and the number of elements
% ====================================================
fprintf('The number of nodes is: %6i\n',Nnods)
fprintf('The number of elements is: %6i\n',Nelms)
%
% Set all triangles in a counter-clockwise sense of numbering
% ===========================================================
for e=1:Nelms
    x21=x(elmnod(e,2))-x(elmnod(e,1));
    x31=x(elmnod(e,3))-x(elmnod(e,1));
    y21=y(elmnod(e,2))-y(elmnod(e,1));
    y31=y(elmnod(e,3))-y(elmnod(e,1));
    Ae=0.5*(x21*y31-x31*y21);
    if Ae < 0
        temp=elmnod(e,2);
        elmnod(e,2)=elmnod(e,3);
        elmnod(e,3)=temp;
    end
end
%
% Plot the mesh
% =============
figure(1)
triplot(elmnod,x,y);
xlabel('x (wavelengths)');
ylabel('y (wavelengths)');
axis([-rho rho -rho rho]);
axis square;
%
% Set the Dirichlet & Absorbing BCs on the cylinder and outer boundary,
% respectively
% =====================================================================
NEBC=0;
NABC=0;
for JJ=1:Nnods
    r0=sqrt(x(JJ)^2+y(JJ)^2);
    if abs(robj-r0) < 0.0001*robj
        NEBC=NEBC+1;
        EBC(NEBC)=JJ;
    elseif abs(rho-r0) < 0.0001*rho
        NABC=NABC+1;
        ABC(NABC)=JJ;
    end
end

for e=1:Nelms
    for I=1:3
        Nd=elmnod(e,I);
        flag(e,I)=0;
        for J=1:NABC
            if Nd == ABC(J)
                flag(e,I)=1;
            end
        end
    end
end 
%
% Print the number of Dirichlet BCs and the number of ABCs
% ========================================================
fprintf('The number of nodes on a Dirichlet boundary is: %6i\n',NEBC)
fprintf('The number of nodes on an ABC boundary is: %6i\n',NABC)
%
% Assign the constants for each element
% =====================================
mur0=1;
epsr0=1;
k0=2*pi;
gamma=k0*j+1/(2*rho);
for e=1:Nelms
    alphax(e)=1/mur0;
    alphay(e)=1/mur0;
    beta(e)=k0^2*epsr0;
    g(e)=0;
end
%
% Initialization of the global K matrix and right-hand side vector
% ================================================================
K=sparse(Nnods,Nnods);
b=zeros(Nnods,1);
%
% Form the element matrices and assemble to the global matrix
% ===========================================================
for e=1:Nelms
    x21=x(elmnod(e,2))-x(elmnod(e,1));
    x31=x(elmnod(e,3))-x(elmnod(e,1));
    x32=x(elmnod(e,3))-x(elmnod(e,2));
    x13=x(elmnod(e,1))-x(elmnod(e,3));
    y12=y(elmnod(e,1))-y(elmnod(e,2));
    y21=y(elmnod(e,2))-y(elmnod(e,1));
    y31=y(elmnod(e,3))-y(elmnod(e,1));
    y23=y(elmnod(e,2))-y(elmnod(e,3));
    Ae=0.5*(x21*y31-x31*y21);
    
    % Evaluation of the element K matrix
    % ==================================
    Me(1,1)=-(alphax(e)*y23^2+alphay(e)*x32^2)/(4*Ae);
    Me(1,2)=-(alphax(e)*y23*y31+alphay(e)*x32*x13)/(4*Ae);
    Me(2,1)=Me(1,2);
    Me(1,3)=-(alphax(e)*y23*y12+alphay(e)*x32*x21)/(4*Ae);
    Me(3,1)=Me(1,3);
    Me(2,2)=-(alphax(e)*y31^2+alphay(e)*x13^2)/(4*Ae);
    Me(2,3)=-(alphax(e)*y31*y12+alphay(e)*x13*x21)/(4*Ae);
    Me(3,2)=Me(2,3);
    Me(3,3)=-(alphax(e)*y12^2+alphay(e)*x21^2)/(4*Ae);
    
    % Evaluation of the element T matrix
    % ==================================
    Te(1,1)=beta(e)*Ae/6;
    Te(1,2)=beta(e)*Ae/12;
    Te(2,1)=beta(e)*Ae/12;
    Te(1,3)=beta(e)*Ae/12;
    Te(3,1)=beta(e)*Ae/12;
    Te(2,2)=beta(e)*Ae/6;
    Te(2,3)=beta(e)*Ae/12;
    Te(3,2)=beta(e)*Ae/12;
    Te(3,3)=beta(e)*Ae/6;
    
    % Sum the element matrices Me and Te
    % ==================================
    for I=1:3
        for J=1:3
            Ke(I,J)=Me(I,J)+Te(I,J);
        end
    end
    
    % Evaluation of element vector ge
    % ===============================
    ge(1)=g(e)*Ae/3;
    ge(2)=g(e)*Ae/3;
    ge(3)=g(e)*Ae/3;
    
    % Evaluation of the element vector pe & update of the element K-matrix
    % ====================================================================
    pe=zeros(3,1);
    if flag(e,1) == 1 && flag(e,2) == 1
        x1=x(elmnod(e,1));
        y1=y(elmnod(e,1));
        x2=x(elmnod(e,2));
        y2=y(elmnod(e,2));
        x21=x2-x1;
        Ledg=sqrt((x2-x1)^2+(y2-y1)^2);
        q0=gamma-j*k0*(y2-y1)/Ledg;
        C0=E0*q0*Ledg*exp(-j*k0*x1);
        if(x21 ~= 0)
            C1=(1-j*k0*x21-exp(-j*k0*x21))/(k0^2*x21^2);
            C2=(-1+j*k0*x21*exp(-j*k0*x21)+exp(-j*k0*x21))/(k0^2*x21^2);
        else
            C1=0.5;
            C2=0.5;
        end
        pe(1)=C0*C1;
        pe(2)=C0*C2;
        pe(3)=0;
        
        Ke(1,1)=Ke(1,1)-gamma*Ledg/3;
        Ke(1,2)=Ke(1,2)-gamma*Ledg/6;
        Ke(2,1)=Ke(2,1)-gamma*Ledg/6;
        Ke(2,2)=Ke(2,2)-gamma*Ledg/3;
    elseif flag(e,1) == 1 && flag(e,3) == 1
        x1=x(elmnod(e,1));
        y1=y(elmnod(e,1));
        x3=x(elmnod(e,3));
        y3=y(elmnod(e,3));
        x13=x1-x3;
        Ledg=sqrt((x1-x3)^2+(y1-y3)^2);
        q0=gamma-j*k0*(y1-y3)/Ledg;
        C0=E0*q0*Ledg*exp(-j*k0*x3);
        if(x13 ~= 0)
            C1=(1-j*k0*x13-exp(-j*k0*x13))/(k0^2*x13^2);
            C2=(-1+j*k0*x13*exp(-j*k0*x13)+exp(-j*k0*x13))/(k0^2*x13^2);
        else
            C1=0.5;
            C2=0.5;
        end
        pe(1)=C0*C2;
        pe(2)=0;
        pe(3)=C0*C1;
        
        Ke(1,1)=Ke(1,1)-gamma*Ledg/3;
        Ke(1,3)=Ke(1,3)-gamma*Ledg/6;
        Ke(3,1)=Ke(3,1)-gamma*Ledg/6;
        Ke(3,3)=Ke(3,3)-gamma*Ledg/3;
    elseif flag(e,2) == 1 && flag(e,3) == 1
        x2=x(elmnod(e,2));
        y2=y(elmnod(e,2));
        x3=x(elmnod(e,3));
        y3=y(elmnod(e,3));
        x32=x3-x2;
        Ledg=sqrt((x3-x2)^2+(y3-y2)^2);
        q0=gamma-j*k0*(y3-y2)/Ledg;
        C0=E0*q0*Ledg*exp(-j*k0*x2);
        if(x32 ~= 0)
            C1=(1-j*k0*x32-exp(-j*k0*x32))/(k0^2*x32^2);
            C2=(-1+j*k0*x32*exp(-j*k0*x32)+exp(-j*k0*x32))/(k0^2*x32^2);
        else
            C1=0.5;
            C2=0.5;
        end
        pe(1)=0;
        pe(2)=C0*C1;
        pe(3)=C0*C2;
        
        Ke(2,2)=Ke(2,2)-gamma*Ledg/3;
        Ke(2,3)=Ke(2,3)-gamma*Ledg/6;
        Ke(3,2)=Ke(3,2)-gamma*Ledg/6;
        Ke(3,3)=Ke(3,3)-gamma*Ledg/3;
    end   
    
    % Assemble element matrices & vectors into the global K matrix and b
    % vector
    % ==================================================================
    for I=1:3
        for J=1:3
            K(elmnod(e,I),elmnod(e,J))=K(elmnod(e,I),elmnod(e,J))+Ke(I,J);
        end
        b(elmnod(e,I))=b(elmnod(e,I))+ge(I)-pe(I);
    end
end
%
% Imposition of Dirichlet boundary conditions
% ===========================================
for I=1:NEBC
    for J=1:Nnods
        if J ~= EBC(I) 
            b(J)=b(J)-K(J,EBC(I))*0;
        end
    end
    K(:,EBC(I))=0;
    K(EBC(I),:)=0;
    K(EBC(I),EBC(I))=1;
    b(EBC(I))=0;
end
%
% Solution of the global matrix system
% ====================================
Ez=K\b;
%
% Generate the solution over a grid and plot it
% =============================================
%[xgrid,ygrid]=meshgrid(-rho:0.0025*(2*rho):rho,-rho:0.0025*(2*rho):rho);
[xgrid,ygrid]=meshgrid(-rho:0.01*(2*rho):rho,-rho:0.01*(2*rho):rho);
Ezgrid=zeros(101,101); %Ezgrid=zeros(401,401);
for I=1:101 %401
    for J=1:101 %401
       % I
       % J
        for e=1:Nelms
         
            x2p=x(elmnod(e,2))-xgrid(I,J);
            x3p=x(elmnod(e,3))-xgrid(I,J);
            y2p=y(elmnod(e,2))-ygrid(I,J);
            y3p=y(elmnod(e,3))-ygrid(I,J);
            A1=0.5*abs(x2p*y3p-x3p*y2p);
                
            x2p=x(elmnod(e,2))-xgrid(I,J);
            x1p=x(elmnod(e,1))-xgrid(I,J);
            y2p=y(elmnod(e,2))-ygrid(I,J);
            y1p=y(elmnod(e,1))-ygrid(I,J);
            A2=0.5*abs(x2p*y1p-x1p*y2p);
                
            x1p=x(elmnod(e,1))-xgrid(I,J);
            x3p=x(elmnod(e,3))-xgrid(I,J);
            y1p=y(elmnod(e,1))-ygrid(I,J);
            y3p=y(elmnod(e,3))-ygrid(I,J);
            A3=0.5*abs(x1p*y3p-x3p*y1p);
                
            x21=x(elmnod(e,2))-x(elmnod(e,1));
            x31=x(elmnod(e,3))-x(elmnod(e,1));
            y21=y(elmnod(e,2))-y(elmnod(e,1));
            y31=y(elmnod(e,3))-y(elmnod(e,1));
            Ae=0.5*(x21*y31-x31*y21);
                
            if abs(Ae-(A1+A2+A3)) < 0.00001*Ae   
                 ksi=(y31*(xgrid(I,J)-x(elmnod(e,1)))-x31*(ygrid(I,J)-y(elmnod(e,1))))/(2*Ae);
                 ita=(-y21*(xgrid(I,J)-x(elmnod(e,1)))+x21*(ygrid(I,J)-y(elmnod(e,1))))/(2*Ae);
                 N1=1-ksi-ita;
                 N2=ksi;
                 N3=ita;
                 Ezgrid(I,J)=N1*Ez(elmnod(e,1))+N2*Ez(elmnod(e,2))+N3*Ez(elmnod(e,3));
            end
        end
    end
end
%
% Plot the total electric field obtained from the finite element solution
% on a contour plot
% =======================================================================
figure(2)
%contour(xgrid,ygrid,abs(Ezgrid),50);
contourf(xgrid,ygrid,abs(Ezgrid));
xlabel('x (wavelengths)');
ylabel('y (wavelengths)');
axis([-rho rho -rho rho]);
axis square;
colorbar;
%
% Evaluate the exact analytical solution at a boundary half-way between the
% scatterer and the ABC boundary
% =========================================================================
Np=50;
d2p=pi/180;
dist=robj+(rho-robj)/2;
Ezexct=zeros(1,1441); %Ezexct=zeros(1,721);
for I=1:1441 %721
    phi(I)=(I-1)*0.25; %0.5;
    xeval=dist*cos(phi(I)*d2p);
    yeval=dist*sin(phi(I)*d2p);
    for e=1:Nelms     
        x2p=x(elmnod(e,2))-xeval;
        x3p=x(elmnod(e,3))-xeval;
        y2p=y(elmnod(e,2))-yeval;
        y3p=y(elmnod(e,3))-yeval;
        A1=0.5*abs(x2p*y3p-x3p*y2p);
                
        x2p=x(elmnod(e,2))-xeval;
        x1p=x(elmnod(e,1))-xeval;
        y2p=y(elmnod(e,2))-yeval;
        y1p=y(elmnod(e,1))-yeval;
        A2=0.5*abs(x2p*y1p-x1p*y2p);
                
        x1p=x(elmnod(e,1))-xeval;
        x3p=x(elmnod(e,3))-xeval;
        y1p=y(elmnod(e,1))-yeval;
        y3p=y(elmnod(e,3))-yeval;
        A3=0.5*abs(x1p*y3p-x3p*y1p);
                
        x21=x(elmnod(e,2))-x(elmnod(e,1));
        x31=x(elmnod(e,3))-x(elmnod(e,1));
        y21=y(elmnod(e,2))-y(elmnod(e,1));
        y31=y(elmnod(e,3))-y(elmnod(e,1));
        Ae=0.5*(x21*y31-x31*y21);
                
        if abs(Ae-(A1+A2+A3)) < 0.00001*Ae   
            ksi=(y31*(xeval-x(elmnod(e,1)))-x31*(yeval-y(elmnod(e,1))))/(2*Ae);
            ita=(-y21*(xeval-x(elmnod(e,1)))+x21*(yeval-y(elmnod(e,1))))/(2*Ae);
            N1=1-ksi-ita;
            N2=ksi;
            N3=ita;
            Ezeval(I)=N1*Ez(elmnod(e,1))+N2*Ez(elmnod(e,2))+N3*Ez(elmnod(e,3));
        end
    end
    for n=-Np:Np
        Ezexct(I)=Ezexct(I)+E0*j^(-n)*(besselj(n,k0*dist)-besselj(n,k0*robj)/besselh(n,2,k0*robj)*besselh(n,2,k0*dist))*exp(j*n*phi(I)*d2p);
    end
end
%
% Plot the exact analytical solution and the finite element solution along
% a boundary half-way between the scatterer and the ABC boundary
% ========================================================================
figure(3)
plot(phi,abs(Ezexct),'k-',phi,abs(Ezeval),'k--'),legend('Exact','FEM');
xlabel('Observation Angle (degrees)');
ylabel('Electric Field (V/m)');
axis([0 360 0 2*E0]);
