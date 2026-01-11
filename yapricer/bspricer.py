import numpy as np
from aleatory.processes import GBM
from scipy.stats import norm
import matplotlib.pyplot as plt

class BSModel:
    def __init__(self, spot, K, T, r, sigma):
        self.S0 = spot      # Initial stock price
        self.K = K        # Strike price
        self.T = T        # Time to maturity
        self.r = r        # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset
        self.GBM = GBM(initial=self.S0, drift=self.r, volatility=self.sigma, T=self.T)

    def d1(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        return d1
    
    def d2(self):
        return self.d1() - self.sigma * self.T ** 0.5
    
    @staticmethod
    def BSFormula(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        elif option_type == 'put':
            price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return price
    
    @staticmethod
    def european_payoff(S_T, K, option_type='call'):
        if option_type == 'call':
            payoff = np.maximum(S_T - K, 0)
        elif option_type == 'put':
            payoff = np.maximum(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        return payoff
    
    def european_price(self, option_type='call'):
        price = self.BSFormula(self.S0, self.K, self.T, self.r, self.sigma, option_type)
        return price
    
    def simulate_price(self, n, N):
        paths = self.GBM.simulate(n=n, N=N)
        return paths
    
    def simulate_terminal_price(self,T, size):
        terminal_distribution = self.GBM.get_marginal(T)
        sample = terminal_distribution.sample(size)
        return sample
    
    @staticmethod
    def BSGreeks(greek, option_type, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if greek == 'delta':
            if option_type == 'call':
                delta = norm.cdf(d1)
            elif option_type == 'put':
                delta = norm.cdf(d1) - 1
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            return delta
        
        elif greek == 'gamma':
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return gamma
        
        elif greek == 'vega':
            vega = S * norm.pdf(d1) * np.sqrt(T)
            return vega
        
        elif greek == 'theta':
            if option_type == 'call':
                theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                          r * K * np.exp(-r * T) * norm.cdf(d2))
            elif option_type == 'put':
                theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                          r * K * np.exp(-r * T) * norm.cdf(-d2))
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            return theta
        
        elif greek == 'rho':
            if option_type == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            elif option_type == 'put':
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            return rho
        
        else:
            raise ValueError("greek must be one of 'delta', 'gamma', 'vega', 'theta', or 'rho'")
        
    
    def delta(self, option_type='call'):
        d1 = self.d1()
        
        if option_type == 'call':
            delta = norm.cdf(d1)
        elif option_type == 'put':
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return delta
    
    def gamma(self):
        d1 = self.d1()
        gamma = norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))
        return gamma
    
    def vega(self):
        d1 = self.d1()
        vega = self.S0 * norm.pdf(d1) * np.sqrt(self.T)
        return vega
    
    def theta(self, option_type='call'):
        d1 = self.d1()
        d2 = self.d2()
        
        if option_type == 'call':
            theta = (- (self.S0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) - 
                      self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        elif option_type == 'put':
            theta = (- (self.S0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) + 
                      self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return theta
    
    
    def rho(self, option_type='call'):
        d2 = self.d2()
        
        if option_type == 'call':
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'put':
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return rho
    
    def plot_european_price_vs_spot(self, option_type='call', spot_range=(50, 150), num_points=100):
        spots = np.linspace(spot_range[0], spot_range[1], num_points)
        prices = [self.BSFormula(S, self.K, self.T, self.r, self.sigma, option_type) for S in spots]
        payoffs = [self.european_payoff(S, self.K, option_type) for S in spots]
        
        plt.plot(spots, prices, label=f'{option_type.capitalize()} Option Price')
        plt.plot(spots, payoffs, label=f'{option_type.capitalize()} Option Payoff', linestyle='--')
        plt.plot(self.K, self.BSFormula(self.K, self.K, self.T, self.r, self.sigma, option_type), 'ro', label='ATM Price')
        plt.xlabel('Spot Price')
        plt.ylabel('Option Price')
        plt.title(f'European {option_type.capitalize()} Option Price vs Spot Price')
        plt.legend()
        plt.grid()
        plt.show()
      
    def plot_greeks_vs_spot(self, greek='delta', spot_range=(50, 150), num_points=100):
        spots = np.linspace(spot_range[0], spot_range[1], num_points)
        greeks = [self.BSGreeks(greek, 'call', S, self.K, self.T, self.r, self.sigma) for S in spots]
        
        plt.plot(spots, greeks, label=f'{greek.capitalize()} (Call)')
        plt.plot(self.K, self.BSGreeks(greek, 'call', self.K, self.K, self.T, self.r, self.sigma), 'ro', label=f'ATM {greek.capitalize()}')
        plt.xlabel('Spot Price')
        plt.ylabel(f'{greek.capitalize()}')
        plt.title(f'European Call Option {greek.capitalize()} vs Spot Price')
        plt.legend()
        plt.grid()
        plt.show()
        
    
    def plot_greeks_vs_strike(self, greek='delta', strike_range=(50, 150), num_points=100):
        strikes = np.linspace(strike_range[0], strike_range[1], num_points)
        greeks = [self.BSGreeks(greek, 'call', self.S0, K, self.T, self.r, self.sigma) for K in strikes]
        
        plt.plot(strikes, greeks, label=f'{greek.capitalize()} (Call)')
        plt.xlabel('Strike Price')
        plt.ylabel(f'{greek.capitalize()}')
        plt.title(f'European Call Option {greek.capitalize()} vs Strike Price')
        plt.legend()
        plt.grid()
        plt.show()
        
if __name__ == "__main__":
    # Example usage
    model = BSModel(spot=100, K=100, T=1.0, r=0.05, sigma=0.2)
    call_price = model.european_price(option_type='call')
    put_price = model.european_price(option_type='put')
    
    print(f'Call Option Price: {call_price}')
    print(f'Put Option Price: {put_price}')
    
    model.plot_european_price_vs_spot(option_type='call')
    model.plot_european_price_vs_spot(option_type='put')
    
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        model.plot_greeks_vs_spot(greek=greek, spot_range=(50, 200))
    
    
