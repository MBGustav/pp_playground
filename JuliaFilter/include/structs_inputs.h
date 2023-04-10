
//Facilitando o uso de complexo
struct Complex
{
    float r,i;
    Complex(float a,float b): r(a), i(b){}
    float mag() {return r*r+i*i;}
    Complex operator*(const Complex &a){
        return Complex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    Complex operator+(const Complex &a){
        return Complex(r+a.r, i+a.i);
    }
};

int julia( int x, int y, int DIM) { 
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    Complex c(-0.8, 0.156);
    Complex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;

        //Verifica Divergencia
        if (a.mag() > 1000)
            return 0;
    }
    return 1;
}

// #ifdef SERIAL

void JuliaFilter( unsigned char *ptr, int DIM ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int Jval = julia( x, y, DIM);
            ptr[offset*3 + 0] = 255 * Jval;
            ptr[offset*3 + 1] = 150;
            ptr[offset*3 + 2] = 0;
        }
    }
 }


// #endif


#ifdef OPENMP
void JuliaFilter( unsigned char *ptr, int DIM ){
    int y,x,offset;
    #pragma omp parallel for private(y, x, offset) shared(ptr)
    for (y=0; y<DIM; y++) {
        for (x=0; x<DIM; x++) {
            offset = x + y * DIM;

            int Jval = julia( x, y, DIM);
            ptr[offset*3 + 0] = 255 * Jval;
            ptr[offset*3 + 1] = 150;
            ptr[offset*3 + 2] = 0;
        }
    }
 }
 #endif
