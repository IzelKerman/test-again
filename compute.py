import numpy as np
import scipy.integrate as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def ell_to_cart(r, theta, phi, a=1):
    rho = np.sqrt(r ** 2 + a ** 2)
    return rho * np.sin(theta) * np.cos(phi), rho * np.sin(theta) * np.sin(phi), r * np.cos(theta)


def cot(x):
    return 1 / np.tan(x)


def inside_event_horizon(t, y):
    r = y[0]
    a = y[5]
    return r - 1 - np.sqrt(1 - a ** 2)
inside_event_horizon.Terminal = True


def g(r, theta, a):
    rho = np.sqrt(r ** 2 + (a * np.cos(theta)) ** 2)
    Delta = r ** 2 - 2 * r + a ** 2
    Sigma = np.sqrt((r ** 2 + a ** 2) ** 2 - a ** 2 * Delta * np.sin(theta) ** 2)
    alpha = rho * np.sqrt(Delta) / Sigma
    omega = 2 * a * r / Sigma ** 2
    Omega = Sigma * np.sin(theta) / rho
    g_00 = (Omega * omega) ** 2 - alpha ** 2
    g_11 = rho ** 2 / Delta
    g_22 = rho ** 2
    g_33 = Omega ** 2
    g_03 = - Omega ** 2 * omega
    return [g_00, g_11, g_22, g_33, g_03]


def g_inv(r, theta):
    pass


def scalar_prod(r, theta, u, v, G=None):
    if G is None:
        g_ = g(r, theta)
    else:
        g_ = G
    return g_[0] * u[0] * v[0] + g_[1] * u[1] * u[1] + g_[2] * u[2] * v[2] + g_[3] * u[3] * v[3] + g_[4] * (u[0] * v[3] + u[3] * v[0])


def flat(r, theta, u, G=None):
    if G is None:
        g_ = g(r, theta)
    else:
        g_ = G
    return np.array([g_[0] * u[0] + g_[4] * u[3], g_[1] * u[1], g_[2] * u[2], g_[4] * u[0] + g_[3] * u[3]])


def sharp(r, theta, u):
    pass


def f_kerr(t, x):
    #x = r, theta, phi, p_r, p_theta, a, b, q
    r, theta, phi, p_r, p_theta, a, b, q = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    rho = np.sqrt(r ** 2 + (a * np.cos(theta)) ** 2)
    Delta = r ** 2 - 2 * r + a ** 2
    P = r ** 2 + a ** 2 - a * b
    R = P ** 2 - Delta * ((b - a) ** 2 + q)
    Theta = q - (b * cot(theta)) ** 2 + (a * np.cos(theta)) ** 2
    drho_r = r / rho
    drho_theta = - a ** 2 * np.cos(theta) * np.sin(theta) / rho
    dDelta_r = 2 * (r - 1)
    dR_r = 4 * r * P - ((b - a) ** 2 + q) * dDelta_r
    dTheta_theta = 2 * cot(theta) * (1 + cot(theta) ** 2) * b ** 2 - 2 * np.cos(theta) * np.sin(theta) * a ** 2

    dr = Delta * p_r / rho ** 2
    dtheta = p_theta / rho ** 2
    #dphi = (a * P / Delta + (b - a) + b * cot(theta) ** 2) / rho ** 2
    dphi = ((a - 2 * b) * P / Delta + b * (1 + cot(theta) ** 2) - a) / rho ** 2
    truc = Delta * p_r ** 2 + p_theta ** 2 - R / Delta - Theta
    dp_r = (truc * drho_r / rho - (p_r ** 2 + R / Delta ** 2) * dDelta_r / 2 + dR_r / (2 * Delta)) / rho ** 2
    dp_theta = (truc * drho_theta / rho + dTheta_theta / 2) / rho ** 2
    """if 0 <= t % 100 <= 0.002:
        print(int(t) / 1000)"""

    return np.array([dr, dtheta, dphi, dp_r, dp_theta, 0, 0, 0])


def f_kerr_backward(t, x):
    f = f_kerr(t, x)
    return np.array([-f[0], -f[1], -f[2], -f[3], -f[4], f[5], -f[6], f[7]])


class System:
    def __init__(self, a, x, fov, im_size, im_name, background):
        self.a = a
        self.x = x
        self.delta_x = fov[0] / 2
        self.delta_y = fov[1] / 2
        self.N_x = im_size[0]
        self.N_y = im_size[1]
        self.L_x = np.tan(self.delta_x)
        self.L_y = np.tan(self.delta_y)
        self.dx = 2 * self.L_x / self.N_x
        self.dy = 2 * self.L_y / self.N_y
        image_background = Image.open(background)
        self.back_pixels = image_background.load()
        self.background_size = image_background.size

    def create_image(self, image_name):
        img = Image.new('RGB', (self.N_x, self.N_y), (0, 0, 0))
        img.save(image_name)

    def compute_angle(self, i, j):
        n = np.array([1, self.L_x * (-1 + (2 * i + 1) / self.N_x), self.L_y * (-1 + (2 * j + 1) / self.N_y)])
        n = n / np.linalg.norm(n)   #composante de la direction spatiale (au signe près) dans la base ON e_r,...
        """theta = np.arctan2(np.sqrt(n[0] ** 2 + n[1] ** 2), -n[2])
        phi = np.arctan2(n[1], n[0])
        if phi > np.pi:
            phi = phi - 2 * np.pi
        I = np.floor((phi + np.pi) * self.background_size[0] / (2 * np.pi))
        J = np.floor(theta * self.background_size[1] / np.pi)
        return self.back_pixels[I, J]"""
        G = g(self.x[0], self.x[1], self.a)
        #Gamma = - np.sqrt(G[3]) * n[2] * (G[0] + 1) / (2 * G[4])
        #Gamma = (- G[4] * n[2] + np.sqrt(G[4] ** 2 * n[2] ** 2 - G[0] * G[3])) / np.sqrt(G[3])
        #p = np.array([1, Gamma * n[0], - Gamma * n[2], Gamma * n[1]])
        p = np.array([1, n[0] / np.sqrt(G[1]), -n[2] / np.sqrt(G[2]), n[1] / np.sqrt(G[3]) - G[4] / G[3]])
        p = flat(self.x[0], self.x[1], p, G)
        p = - p / p[0]
        b = p[3]
        q = p[2] ** 2 + cot(self.x[1]) ** 2 * b ** 2 - self.a ** 2 * np.cos(self.x[1]) ** 2
        sol = sc.solve_ivp(f_kerr, [0, -10000], [self.x[0], self.x[1], self.x[2], p[1], p[2], self.a, b, q], rtol=1e-9)
        """fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = [ell_to_cart(sol.y[0][i], sol.y[1][i], sol.y[2][i], self.a) for i, y_i in enumerate(sol.y[0])]
        x = [x_i[0] for x_i in X]
        y = [x_i[1] for x_i in X]
        z = [x_i[2] for x_i in X]
        ax.set_xlim3d(-50, 50)
        ax.set_ylim3d(-50, 50)
        ax.set_zlim3d(-50, 50)
        ax.plot(x, y, z)
        plt.show()"""

        if sol.y[0][-1] <= 1000:
            return (0, 0, 0)
        else:
            # on est loin donc ellipsoid approx cartésien
            #theta = sol.y[1][-1] % (np.pi)  # ici c'est faux
            if sol.y[1][-1] % (2 * np.pi) <= np.pi:
                theta = sol.y[1][-1] % np.pi
                #phi = sol.y[2][-1] % (2 * np.pi)
                phi = (np.pi - sol.y[2][-1]) % (2 * np.pi)
            else:
                theta = (2 * np.pi - sol.y[1][-1]) % np.pi
                #phi = (sol.y[2][-1] + np.pi) % (2 * np.pi)
                phi = (-sol.y[2][-1]) % (2 * np.pi)
            phi = sol.y[2][-1] % (2 * np.pi)
            if phi > np.pi:
                phi = phi - 2 * np.pi
            I = np.floor((phi + np.pi) * self.background_size[0] / (2 * np.pi))
            J = np.floor(theta * self.background_size[1] / np.pi)
            try:
                return self.back_pixels[I, J]
            except:
                print("FUCK", I, J)
            #tout est décallé parce que je suis trop con ptn de bordel de merde


if __name__ == "__main__":
    a = 0.5
    b = 0
    q = 0
    r = 10
    theta = np.pi / 2
    phi = 0
    p_r = 0
    p_theta = 0

    rho = np.sqrt(r ** 2 + (a * np.cos(theta)) ** 2)
    Delta = r ** 2 - 2 * r + a ** 2
    Sigma = np.sqrt((r ** 2 + a ** 2) ** 2 - a ** 2 * Delta * np.sin(theta) ** 2)
    alpha = rho * np.sqrt(Delta) / Sigma
    omega = 2 * a * r / Sigma ** 2
    Omega = Sigma * np.sin(theta) / rho

    print(omega, alpha, Omega)
    b = 1 / (omega + alpha / Omega)
    #b = (r ** 2 + a ** 2 - a * b) / np.sqrt(Delta) + a

    plt.style.use('dark_background')

    sol = sc.solve_ivp(f_kerr, [0, 1125], [r, theta, phi, p_r, p_theta, a, b, q])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = [ell_to_cart(sol.y[0][i], sol.y[1][i], sol.y[2][i], a) for i, y_i in enumerate(sol.y[0])]
    x = [x_i[0] for x_i in X]
    y = [x_i[1] for x_i in X]
    z = [x_i[2] for x_i in X]
    ax.plot(x, y, z)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.plot(x, y)
    plt.show()


    #mayble funny, well no but just a bit : b = 8, q = 1, p_theta = 1, p_r = 0, r = 10, a = 1, t = 1125
