/* Same fixed-coefficient F dynamics as physics.py. RK4 + cubic Hermite events.
   No inference, data generation, calibration or scoring in this file. */
#include <math.h>
#include <stddef.h>

static void rhs(const double *s, double by, double bz, const double *e, double *d) {
    double ux=s[3]-e[2], uy=s[4]-e[3], uz=s[5]-e[4];
    double f=e[0]*sqrt(ux*ux+uy*uy+uz*uz), cd=e[5];
    d[0]=s[3]; d[1]=s[4]; d[2]=s[5];
    d[3]=f*(-cd*ux+by*uz-bz*uy);
    d[4]=f*(-cd*uy+bz*ux);
    d[5]=-e[1]+f*(-cd*uz-by*ux);
}
static void step(const double *a,double by,double bz,const double *e,double h,double *b) {
    double k1[6],k2[6],k3[6],k4[6],s[6];
    rhs(a,by,bz,e,k1);
    for(int j=0;j<6;j++)s[j]=a[j]+h*.5*k1[j]; rhs(s,by,bz,e,k2);
    for(int j=0;j<6;j++)s[j]=a[j]+h*.5*k2[j]; rhs(s,by,bz,e,k3);
    for(int j=0;j<6;j++)s[j]=a[j]+h*k3[j]; rhs(s,by,bz,e,k4);
    for(int j=0;j<6;j++)b[j]=a[j]+h*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6;
}
static double pos(const double *a,const double *b,int j,double h,double u) {
    double u2=u*u,u3=u2*u;
    return (2*u3-3*u2+1)*a[j]+(u3-2*u2+u)*h*a[j+3]+(-2*u3+3*u2)*b[j]+(u3-u2)*h*b[j+3];
}
static double root(const double *a,const double *b,int j,double target,double h,int up) {
    double lo=0,hi=1;
    for(int k=0;k<38;k++) {
        double m=(lo+hi)*.5,v=pos(a,b,j,h,m)-target;
        if((up && v<0)||(!up && v>0))lo=m;else hi=m;
    }
    return (lo+hi)*.5;
}
/* outputs: obs n*6; event codes 1 inside,2 outside,3 ground,4 boundary,5 unresolved;
   event_yzt n*3. geom: initial xyz, goalx, radius,ylo,yhi,zlo,zhi,horizon. */
void propagate_batch(size_t n,const double *par,const double *env,const double *geom,
                     double dt,double *obs,int *event,double *yzt) {
    for(size_t i=0;i<n;i++) {
        const double *p=par+5*i;
        double a[6]={geom[0],geom[1],geom[2],p[0],p[1],p[2]},b[6],t=0;
        for(int j=0;j<6;j++)obs[6*i+j]=NAN;
        for(int j=0;j<3;j++)yzt[3*i+j]=NAN;
        event[i]=5;
        if(a[2]<=geom[4] && a[5]<=0){event[i]=3;continue;}
        int frame=0;
        while(t<geom[9]-1e-12) {
            double h=fmin(dt,geom[9]-t);
            if(frame<2 && t+h>.05*(frame+1))h=.05*(frame+1)-t;
            step(a,p[3],p[4],env,h,b);
            int gx=(a[0]<geom[3] && b[0]>=geom[3]),gz=(b[2]<=geom[4] && b[5]<0);
            if(gx||gz) {
                double x=gx?root(a,b,0,geom[3],h,1):2,
                       z=gz?root(a,b,2,geom[4],h,0):2,u=fmin(x,z);
                double px=pos(a,b,0,h,u),py=pos(a,b,1,h,u),pz=pos(a,b,2,h,u);
                int boundary=(fabs(x-z)*h<=1e-7) ||
                    (x<z && b[5]<0 && fabs(pz-geom[4])<=fabs(b[5])*1e-7+1e-8) ||
                    (z<x && b[3]>0 && fabs(px-geom[3])<=fabs(b[3])*1e-7+1e-8);
                event[i]=boundary?4:(x<z?((py>geom[5]&&py<geom[6]&&pz>geom[7]&&pz<geom[8])?1:2):3);
                yzt[3*i]=py;yzt[3*i+1]=pz;yzt[3*i+2]=t+h*u;
                break;
            }
            t+=h;
            for(int j=0;j<6;j++)a[j]=b[j];
            if(frame<2 && fabs(t-.05*(frame+1))<1e-10) {
                for(int j=0;j<3;j++)obs[6*i+3*frame+j]=a[j];
                frame++;
            }
        }
    }
}
