import numpy as np
import matplotlib.pyplot as plt
import random
import time as tm
#from mpl_toolkits import mplot3d

start_time=tm.time()

total_time=4.0 # In seconds (must be multiple of 4)
time_step=0.01 # In seconds

Nx=9
Ny=10
N=Nx*Ny

sigma=0.24
I=0.3
T=0.05
tau=0.8

time=np.arange(0,total_time,time_step)
neuro=np.arange(1,(Nx*Ny)+1)

#Initial Activity of neurons
A=np.random.uniform(0,1/np.sqrt(N),[N,int(total_time/time_step)]) #Randomly initialized states of the neurons between 0 - 1/sqrt(N)
np.savetxt('A',A)
A=np.loadtxt('A')

#Weight matrix
W=np.zeros((N,N))

#theta - phi for constant eye movements
theta_list=np.zeros((1,int(total_time/time_step)))
phi_list=np.zeros((1,int(total_time/time_step)))
theta_temp=0
phi_temp=0
sweep_no=1
sweep_time=total_time/sweep_no


def coordinates(num):
    
    if num<=(Nx-1):
        x=num+1
        y=1
    else:
        x=(num%Nx)+1
        y=int(num/Nx)+1

    x=(x-0.5)/Nx
    y=(0.866*(y-0.5))/Ny
    
    return x,y
    
def coordinates_int(num):
    if num<=(Nx-1):
        x=num+1
        y=1
    else:
        x=(num%Nx)+1
        y=int(num/Nx)+1
    
    return x,y
    
def distances(ix,iy,jx,jy,mx,my):
    s=np.array([[0,0],[-0.5,np.sqrt(3)/2],[-0.5,-np.sqrt(3)/2],[0.5,np.sqrt(3)/2],[0.5,-np.sqrt(3)/2],[-1,0],[1,0]])
    d_min=99999
        
    for j in range(7):
        d_new=(ix-jx+s[j,0]+mx)**2 + (iy-jy+s[j,1]+my)**2
        
        if d_new<d_min:
            d_min=d_new
    
    return d_min
        
def update_weights(W1,mx,my):
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            

            ix,iy=coordinates(i)
            jx,jy=coordinates(j)
            
            
            d=distances(ix,iy,jx,jy,mx,my)
            
        
            W1[i,j]=(I*np.exp((-d)/(sigma**2)))-T
            
    return W1
    
def activation(tau,At,W2):
    At_next=At

    for m in range(At.shape[0]):
        temp1=At[m]+np.sum(np.multiply(At,W2[m,:]))
        temp2=temp1+(tau*((temp1/np.average(At))-temp1))
        
        
        if temp2 < 0:
            temp2=0
        At_next[m]=temp2
        
        
    return At_next

def randomwalk2D(n,dx,dy):
    
    x = np.zeros(n)
    y = np.zeros(n)
    
    directions = ["UP", "DOWN", "LEFT", "RIGHT", "Diag_Left_Up", "Diag_Left_Down", "Diag_Right_Up", "Diag_Right_Down"]
    for i in range(1, n):
        # Pick a direction at random
        step = random.choice(directions)

        # Move the object according to the direction
        if step == "RIGHT":
            x[i] = x[i - 1] + dx
            y[i] = y[i - 1]
        elif step == "LEFT":
            x[i] = x[i - 1] - dy
            y[i] = y[i - 1]
        elif step == "UP":
            x[i] = x[i - 1]
            y[i] = y[i - 1] + dy
        elif step == "DOWN":
            x[i] = x[i - 1]
            y[i] = y[i - 1] - dx
        elif step == "Diag_Left_Up":
            x[i] = x[i - 1] - dx
            y[i] = y[i - 1] + dy
        elif step == "Diag_Left_Down":
            x[i] = x[i - 1] - dx
            y[i] = y[i - 1] - dy
        elif step == "Diag_Right_Up":
            x[i] = x[i - 1] + dx
            y[i] = y[i - 1] + dy
        elif step == "Diag_Right_Down":
            x[i] = x[i - 1] + dx
            y[i] = y[i - 1] - dy
        
    # making positive coordinates
    if np.min(x)<0:
        bias=-np.min(x)
        x = x + bias
        
    if np.min(y)<0:
        bias1=-np.min(y)
        y = y + bias1
    
    return x, y



def periodic_boundary_weights(W1,k,theta,phi):
    
    for test in range(sweep_no):
        if test % 2 != 0:
            if k > ((sweep_time*test)/time_step) and k <= (((sweep_time*(test+1))/time_step)):
                phi=-phi
                print(k)
    
    global theta_temp, phi_temp
    
    
    W2=update_weights(W1,theta,phi)
    theta_list[0,k]=theta_temp+theta
    phi_list[0,k]=phi_temp+phi
    
    theta_temp=theta_list[0,k]
    phi_temp=phi_list[0,k]
    #print(theta_temp,phi_temp)
    
    return W2


def calculate_respone():
    A_temp=A
    

    
    for k in range(A_temp.shape[1]-1):
        
        #for random eye movement
        #W_new=update_weights(W,theta_list[0,k+1]-theta_list[0,k],phi_list[0,k+1]-phi_list[0,k])
        
        #for constant eye movement
        #W_new=periodic_boundary_weights(W,k+1,1.57/((sweep_time*sweep_no)/time_step),1.57/(sweep_time/time_step))    
        
        #Eye Fixation
        W_new=update_weights(W,0,0)
        
        #Current state
        At=A_temp[:,k]
        
        #Next State
        At_next=activation(tau,At,W_new)
        #dif=At_next-At
        #print(dif)
        A_temp[:,k+1]=At_next
        
        print ("\rFor I =",I,"--",int(100*k/(A_temp.shape[1]-1)),"%",end='',flush=True)
        
    return A_temp

def plot_and_save(Ab):

    my_dpi=200
    plt.figure(figsize=(2000/my_dpi, 1600/my_dpi), dpi=my_dpi)
    #plt.imshow(A,cmap='hot', interpolation='none',aspect='auto')
    plt.pcolor(time,neuro,Ab/np.max(Ab))    
    plt.title('I=%f' % I)
    plt.xlabel("Time (in seconds)",fontsize=13, fontweight='bold')
    plt.ylabel("Neuron ID",fontsize=13, fontweight='bold')
    plt.colorbar()
    tt=1.57/((sweep_time*sweep_no)/time_step)
    pp=1.57/((sweep_time)/time_step)
    plt.savefig("activity_theta%f_phi%f _I%f.png" % (tt, pp, I),dpi=my_dpi)
    
    plt.figure(figsize=(1800/my_dpi, 1200/my_dpi), dpi=my_dpi)
    plt.title("Acativation of Neuron 47 under I=%f - Eye movement (d(theta)/dt=%f, d(phi)/dt=%f" %(I, tt, pp))
    plt.xlabel("Time (in seconds)",fontsize=13, fontweight='bold')
    plt.ylabel("Activation Level",fontsize=13, fontweight='bold')
    plt.plot(time,A[47,:])
    plt.savefig('activity_N47-theta%f_phi%f _I%f.png' % (tt, pp, I),dpi=my_dpi)
    
    plt.figure(figsize=(1800/my_dpi, 1200/my_dpi), dpi=my_dpi)
    plt.title("Acativation of Neuron 79 under I=%f - Eye movement (d(theta)/dt=%f, d(phi)/dt=%f" %(I, tt, pp))
    plt.xlabel("Time (in seconds)",fontsize=13, fontweight='bold')
    plt.ylabel("Activation Level",fontsize=13, fontweight='bold')
    plt.plot(A[79,:])
    plt.savefig('activity_N79--theta%f_phi%f _I%f.png' % (tt, pp, I),dpi=my_dpi)
        
    plt.show()
    

###### main ######
def main():
    
    start_I=15.0
    stop_I=46.0
    step_I=1
    baseline_I=15.0
    total_I=int(((stop_I-start_I)/step_I))
    I_list=(np.arange(start_I,stop_I,step_I)-baseline_I)/100
    activity_N47=np.zeros((total_I,int(total_time/time_step))) 
    time_47=np.zeros((total_I,N))
    temp=0
    
    #for random movement
    ######
    # theta, phi=randomwalk2D(int(total_time/time_step),1,1)
    # theta=(1.57*theta)/np.max(theta)  #scaling between 0 to 1.57 (pi/2)
    # phi=(1.57*phi)/np.max(phi) #scaling between 0 to 1.57 (pi/2)
    # global theta_list, phi_list
    # theta_list[0,:]=theta
    # phi_list[0,:]=phi
    ########
    
    print(total_I)
    for ii in np.arange(start_I,stop_I,step_I):
        global I
        I=ii/100
        global theta_temp, phi_temp
        theta_temp=0
        phi_temp=0
        A1=calculate_respone()
        plot_and_save(A1)
        activity_N47[temp,:]=A1[47,:]/np.max(A1[47,:])
        time_47[temp,:]=A1[:,100]/np.max(A1[:,100])
        temp=temp+1
        #print("I =",I)
        
    plt.figure(figsize=(1600/200, 1200/200), dpi=200)
    plt.pcolor(time,I_list/100,activity_N47)
    plt.title("Neuron 47 Activity",fontsize=13)
    plt.xlabel("time (in seconds)",fontsize=13, fontweight='bold')
    plt.ylabel("\u0394I",fontsize=13, fontweight='bold')
    plt.colorbar()
    plt.savefig('Neuron-47-activity-complete-I.png',dpi=500)
    
    plt.figure(figsize=(1600/200, 1200/200), dpi=200)
    plt.pcolor(neuro,I_list/100,time_47)
    plt.title("Activity at Particular Time",fontsize=13)
    plt.xlabel("Nodes",fontsize=13, fontweight='bold')
    plt.ylabel("\u0394I",fontsize=13, fontweight='bold')
    plt.colorbar()
    plt.savefig('snapshot_at_moment.png',dpi=500)
    
    # plt.figure(figsize=(1600/200, 1200/200), dpi=200)
    # plt.pcolor(time,neuro,A1/np.max(A1))
    # plt.title("No Drug (Random Eye Movements)",fontsize=13)
    # plt.xlabel("time (in seconds)",fontsize=13, fontweight='bold')
    # plt.ylabel("Nodes",fontsize=13, fontweight='bold')
    # plt.colorbar()
    # plt.savefig('No-drug-eye-random.png',dpi=500)
    
    # plt.figure(figsize=(1600/200, 1200/200), dpi=200)
    # plt.plot(theta_list[0,:],phi_list[0,:])
    # plt.title("With Drug (Constant Eye Movement)",fontsize=13)
    # plt.xlabel("theta (in radians)",fontsize=13, fontweight='bold')
    # plt.ylabel("phi (in radians)",fontsize=13, fontweight='bold')
    # plt.savefig('eye-constant__theta_phi.png',dpi=500)
    
    plt.show()
    
    end_time=tm.time()
    print("\n---execution time: %s seconds ---" % int(end_time - start_time))
    
if __name__=="__main__":
    main()
