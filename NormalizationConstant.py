from Proximity_functions import *
from scipy.integrate import quad



#------------------------------------------------------------------#
#           LANGER modified centrifugal barrier
#------------------------------------------------------------------#


def V_langer( r , l , Z1 , A1 ,  Z2 , A2 ):
    mu = reduced_mass( Z1 , A1 , Z2 - Z1 , A2 - A1 )
    hbar2 = 197.3269631**2
    return hbar2 * ( l + 0.5 )**2 / ( mu * r**2 )

V_langer_array = np.vectorize( V_langer )


#------------------------------------------------------------------#
#                  LOCAL-EQUIVALENT POTENTIAL
#------------------------------------------------------------------#

# Nonlocality range parameter

beta = 0.22   # fm

def function( x , r , lamb , Z1 , A1 ,  Z2 , A2 ):
    
    # Constant = mu * beta^2 / ( 2 * hbar^2 )
    constant =  reduced_mass( Z1 , A1 , Z2 - Z1 , A2 - A1 ) * beta**2 / ( 2 * 197.3269631**2 )
    
    V = lamb * VN_nl( r , Z1 , A1 , Z2 , A2 )
    Q = Q_value( Z1 , A1 , Z2 , A2 )
            
    arg = float( constant * ( Q - x ) )
    return x * np.exp( arg ) - V

#------------------------------//---------------------------------#

def der_function( x , r , lamb , Z1 , A1 ,  Z2 , A2 ):
    # Constant = mu * beta^2 / ( 2 * hbar^2 )
    constant =  reduced_mass( Z1 , A1 , Z2 - Z1 , A2 - A1 ) * beta**2 / ( 2 * 197.3269631**2 )     

    q_value = Q_value( Z1 , A1 , Z2 , A2 )
    
    arg = float( constant * ( q_value - x ) )
    return np.exp( arg ) - constant * x * np.exp( arg )

#------------------------------//---------------------------------#


def VN_leq( r , lamb , Z1 , A1 ,  Z2 , A2 ):
    root = optimize.newton(function, -50 , fprime = der_function , args = ( r , lamb , Z1 , A1 ,  Z2 , A2 ) )
    return root

VN_leq_array = np.vectorize( VN_leq )


#------------------------------------------------------------------#
#                                 k(r) , W , P 
#                       for non-local potential
#------------------------------------------------------------------#


def turning_function( r , lamb , Z1 , A1 ,  Z2 , A2 ):
    V = lamb * VN_nl( r , Z1 , A1 , Z2 , A2 ) + V_coulomb( r , Z1 , A1 , Z2 , A2 )
    Q = Q_value( Z1 , A1 , Z2 , A2 )
    dif =  float( abs( V - Q ) )
    return dif



def turning_point( start , lamb , Z1 , A1 ,  Z2 , A2 ):
    zero = optimize.fsolve( turning_function , start , args = ( lamb , Z1 , A1 ,  Z2 , A2 ) )[0]
    return zero

turning_point_array = np.vectorize( turning_point )


#------------------------------//---------------------------------#


def k_function( r , lamb , Z1 , A1 ,  Z2 , A2 ):
    m_red = reduced_mass( Z1 , A1 , Z2 , A2 )
    V = lamb * VN_nl( r , Z1 , A1 , Z2 , A2 ) + V_coulomb( r , Z1 , A1 , Z2 , A2 )
    Q = Q_value( Z1 , A1 , Z2 , A2 )
    dif =  float( abs( V - Q ) )
    return np.sqrt( 2 * m_red * dif / ( 197.3269631**2 ) )

def inverse_k( r , lamb , Z1 , A1 ,  Z2 , A2 ):
    return float( 1 / k_function( r , lamb , Z1 , A1 ,  Z2 , A2 ) )


#------------------------------//---------------------------------#


def normalization( r1 , r2 , lamb , Z1 , A1 ,  Z2 , A2 ):
    a = turning_point( r1 , lamb , Z1 , A1 ,  Z2 , A2 )
    b = turning_point( r2 , lamb , Z1 , A1 ,  Z2 , A2 )
    W = quad( inverse_k , a , b , args = ( lamb , Z1 , A1 ,  Z2 , A2 ) )[0]
    return 1 / W


#------------------------------//---------------------------------#


def W_int( r1 , r2 , lamb , Z1 , A1 ,  Z2 , A2 ):
    a = turning_point( r1 , lamb , Z1 , A1 ,  Z2 , A2 )
    b = turning_point( r2 , lamb , Z1 , A1 ,  Z2 , A2 )
    W = quad( k_function , a , b , args = ( lamb , Z1 , A1 ,  Z2 , A2 ) )[0]
    return W


#------------------------------//---------------------------------#


def Borh_Sommerfeld( lamb , r1 , r2 , Z1 , A1 ,  Z2 , A2 ):
    N = A2 - Z2
    if N <= 82:
        G = 18
    elif N > 82 and N <= 126:
        G = 20
    elif N > 126:
        G = 22    
    BS = W_int( r1 , r2 , lamb , Z1 , A1 ,  Z2 , A2 ) - 0.5 * G * np.pi
    return BS


def penetration_prob( r1 , r2 , lamb , Z1 , A1 ,  Z2 , A2 ):
    W = W_int( r1 , r2 , lamb , Z1 , A1 ,  Z2 , A2 )
    return np.exp( -2 * W )


#------------------------------//---------------------------------#


def PD( r3 , lamb , Z1 , A1 ,  Z2 , A2 ):
    Ppp = penetration_prob( r3 , lamb , Z1 , A1 ,  Z2 , A2 )
    Pleq = penetration_prob( r3 , lamb , Z1 , A1 ,  Z2 , A2 )
    return abs( Pleq - Ppp ) * 100 / Pleq




#------------------------------------------------------------------#
#                                 k(r) , W , P 
#                      for Local-equivalent potential
#------------------------------------------------------------------#



def dif_leqQ( r , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 ):
    V = lamb_leq * VN_leq( r , lamb_nl , Z1 , A1 , Z2 , A2 ) + V_coulomb( r , Z1 , A1 , Z2 , A2 )
    Q = Q_value( Z1 , A1 , Z2 , A2 )
    dif =  float( abs( V - Q ) )
    return dif


def zerodif_leq( start , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 ):
    zero = optimize.fsolve( dif_leqQ , start , args = ( lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 ) )[0]
    return zero

def k_leq( r , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 ):
    m_red = reduced_mass( Z1 , A1 , Z2 - Z1 , A2 - A1 )
    V = lamb_leq * VN_leq( r , lamb_nl , Z1 , A1 , Z2 , A2 ) + V_coulomb( r , Z1 , A1 , Z2 , A2 )
    Q = Q_value( Z1 , A1 , Z2 , A2 )
    dif =  float( abs( V - Q ) )
    return np.sqrt( 2 * m_red * dif / ( 197.3269631**2 ) )


def W_leq( r1 , r2 , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 ):
    a = zerodif_leq( r1 , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 )
    b = zerodif_leq( r2 , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 )
    W = quad( k_leq , a , b , args = ( lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 ) )[0]
    return W


def BS_leq( lamb_leq , lamb_nl , r1 , r2 , Z1 , A1 ,  Z2 , A2 ):
    N = A2 - Z2
    if N <= 82:
        G = 18
    elif N > 82 and N <= 126:
        G = 20
    elif N > 126:
        G = 22
    BS = W_leq( r1 , r2 , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 ) - 0.5 * G * np.pi
    return BS


def P_leq( r1_leq , r2_leq , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 ):
    a = zerodif_leq( r1_leq , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 )
    b = zerodif_leq( r2_leq , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 )
    W = W_leq( a , b , lamb_leq , lamb_nl , Z1 , A1 ,  Z2 , A2 )
    return np.exp( -2 * W )


#------------------------------------------------------------------#
#------------------------------//---------------------------------#


