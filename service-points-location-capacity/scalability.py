"""
Scalability test for the SP Location and Capacity model
Based on Raviv (2023) - Transportation Research Part E
Uses real data from Vienna, Graz, and Linz from OpenStreetMap
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.sp_model import ServicePointModel
from models.deterministic_model import DeterministicModel
from models.scenarios_model import ScenariosModel
from utils.osm_austria import load_austria_instance


class ScalabilityTester:
   
    
    def __init__(self, output_dir="data/output/scalability"):
        self.output_dir = output_dir
        self.results = []
    
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
    def create_test_instances_from_cities(self):
        """
        The paper uses:
            - Vienna: 17,701 demand points, 4,260 SP candidates
            - Graz: 6,777 demand points, 1,459 SP candidates
            - Linz: 3,351 demand points, 658 SP candidates
        
        For scalability, we create increasing instances (10%, 20%, ..., 100%)
        """
        
        instances = []
        
        print("Loading real data from Austrian cities:")
        print("-" * 60)
        
        # For each city, create increasing instances
        cities = ['Linz', 'Graz', 'Vienna']  # In order of size
        percentages = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # Increasing percentages
        
        for city in cities:
            print(f"\nLoading {city}...")
            try:
                # Load complete city data - let load_austria_instance handle the file
                full_instance = load_austria_instance(city)
                
                # Create instances of increasing size
                for pct in percentages:
                    # Calculate how many points to include
                    n_demand = int(full_instance['n_demand_points'] * pct)
                    n_sp = int(full_instance['n_sp_locations'] * pct)
                    
                    # Skip instances that are too small
                    if n_demand < 100:
                        continue
                    
                    # Create sub-instance
                    instance = {
                        'name': f"{city}_{int(pct*100)}%",
                        'city': city,
                        'percentage': pct,
                        'demand_points': full_instance['demand_points'][:n_demand],
                        'demand_rates': full_instance['demand_rates'][:n_demand],
                        'sp_locations': full_instance['sp_locations'][:n_sp],
                        'n_demand_points': n_demand,
                        'n_sp_locations': n_sp,
                        'total_demand': sum(full_instance['demand_rates'][:n_demand]),
                        'service_radius': 300,  # 300m for real instances
                        'time_limit': min(300 + n_demand // 10, 1800),  # Proportional time
                        'type': 'real'
                    }
                    
                    instances.append(instance)
                    print(f"  - {instance['name']}: n={n_demand}, m={n_sp}")
                    
            except Exception as e:
                print(f"  Error loading {city}: {e}")
                continue
        
        # Sort by size
        instances.sort(key=lambda x: x['n_demand_points'])
        
        return instances
    
    def test_instance(self, instance):
        """Test an instance with the main stochastic model"""
        
        print(f"\n{'='*60}")
        print(f"Testing: {instance['name']}")
        print(f"  n={instance['n_demand_points']} demand points")
        print(f"  m={instance['n_sp_locations']} SP candidates")
        print('='*60)
        
        # Parameters from the paper
        params = {
            'service_radius': instance.get('service_radius', 300),
            'pickup_probability': 0.5,
            'rejection_cost': 10,  # α = 10 as in main test in the paper
            'setup_base_cost': 10,
            'setup_var_cost': 1/6,
            'n_breakpoints': 20 if instance['n_demand_points'] > 5000 else 15 if instance['n_demand_points'] > 1000 else 12
        }
        
        capacities = [30, 60, 90]  # S = {30, 60, 90} from the paper
        
        # Test only stochastic model for scalability
        print("Solving with STOCHASTIC model (Raviv 2023)...")
        start_total = time.time()
        
        # Build time
        start_build = time.time()
        model = ServicePointModel(
            instance["demand_points"],
            instance["sp_locations"],
            capacities,
            instance["demand_rates"],
            params
        )
        model.build_model()
        build_time = time.time() - start_build
        
        # Solve time
        start_solve = time.time()
        solution = model.solve(
            time_limit=instance['time_limit'],
            mip_gap=0.02 if instance['n_demand_points'] > 5000 else 0.01
        )
        solve_time = time.time() - start_solve
        total_time = time.time() - start_total
        
        # Results
        result = {
            'instance': instance['name'],
            'city': instance.get('city', 'synthetic'),
            'n': instance['n_demand_points'],
            'm': instance['n_sp_locations'],
            'build_time': build_time,
            'solve_time': solve_time,
            'total_time': total_time,
            'objective': solution.get('objective_value', None),
            'mip_gap': solution.get('mip_gap', None),
            'n_sps_opened': solution.get('summary', {}).get('num_service_points', 0),
            'node_count': solution.get('node_count', 0),
            'status': solution.get('status', 'unknown')
        }
        
        print(f"\nResults:")
        print(f"  Build time: {build_time:.2f}s")
        print(f"  Solve time: {solve_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        if result['objective']:
            print(f"  Optimal cost: ${result['objective']:,.0f}")
            print(f"  Open SPs: {result['n_sps_opened']}")
            print(f"  MIP gap: {result['mip_gap']:.2%}")
        
        self.results.append(result)
        self.save_results()
        
        return result
    
    def run_scalability_test(self):
        """Run the complete scalability test with real data"""
        
        print("\n" + "="*70)
        print("SCALABILITY TEST - SP Location and Capacity Problem")
        print("Based on: Tal Raviv (2023) - Transportation Research Part E")
        print("Real data from: Vienna, Graz, Linz (OpenStreetMap)")
        print("="*70)
        
        # Create instances from cities
        instances = self.create_test_instances_from_cities()
        
        if not instances:
            print("\nNo instances created! Check the OSM files.")
            print("Required files:")
            print("  - data/output/osm/vienna.osm.pbf")
            print("  - data/output/osm/graz.osm.pbf")
            print("  - data/output/osm/linz.osm.pbf")
            return
        
        print(f"\n\nTESTING {len(instances)} REAL INSTANCES")
        print("="*70)
        
        # Test each instance
        for instance in instances:
            try:
                self.test_instance(instance)
            except Exception as e:
                print(f"\nError testing {instance['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final analysis
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze scalability results as in the paper"""
        
        print("\n\n" + "="*70)
        print("SCALABILITY ANALYSIS")
        print("="*70)
        
        if not self.results:
            print("No results to analyze!")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(self.results)
        df = df[df['status'] != 'infeasible']  # Remove infeasible
        
        # Print results table
        print("\nDETAILED RESULTS:")
        print("-"*100)
        print(f"{'Instance':<20} {'n':<8} {'m':<8} {'Build(s)':<10} {'Solve(s)':<10} {'Total(s)':<10} {'Cost($)':<12} {'SP':<5}")
        print("-"*100)
        
        for _, row in df.iterrows():
            print(f"{row['instance']:<20} {row['n']:<8} {row['m']:<8} "
                  f"{row['build_time']:<10.2f} {row['solve_time']:<10.2f} "
                  f"{row['total_time']:<10.2f} {row['objective']:<12,.0f} {row['n_sps_opened']:<5}")
        
        # Analysis by city
        print("\n\nANALYSIS BY CITY:")
        print("-"*60)
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            print(f"\n{city}:")
            print(f"  Instances tested: {len(city_data)}")
            print(f"  Range n: {city_data['n'].min()} - {city_data['n'].max()}")
            print(f"  Average time: {city_data['total_time'].mean():.2f}s")
            print(f"  Max time: {city_data['total_time'].max():.2f}s")
        
        # Create plots
        self.create_scalability_plots(df)
        
        # Computational complexity analysis
        
        print("-"*40)
        
        print("THEORETICAL COMPLEXITY:")
        print("-"*40)
        print("This is a Mixed Integer Linear Programming (MILP) problem")
        print("Problem structure:")
        print(f"  - Binary variables: O(m × |S|) where m=SP candidates, |S|=capacities")
        print(f"  - Continuous variables: O(n × m) for assignments + O(m × |S| × K) for PWL")
        print(f"  - Total variables: ~O(n × m)")
        print("\nComplexity classes:")
        print("  - Problem type: NP-hard (facility location + capacity constraints)")
        print("  - Worst-case complexity: O(2^(m×|S|)) - EXPONENTIAL")
        
        
    
        
        # Polynomial fit to estimate empirical behavior (NOT true complexity)
        if len(df) >= 5:
            # Calculate empirical scaling
            log_n = np.log(df['n'].values)
            log_time = np.log(df['total_time'].values)
            coeffs_n = np.polyfit(log_n, log_time, 1)
            
            # Consider both n and m
            problem_size = df['n'].values * df['m'].values
            log_size = np.log(problem_size)
            coeffs_nm = np.polyfit(log_size, log_time, 1)
            
            # Large instances only
            df_large = df[df['n'] > 1000]
            if len(df_large) >= 5:
                log_n_large = np.log(df_large['n'].values)
                log_time_large = np.log(df_large['total_time'].values)
                coeffs_n_large = np.polyfit(log_n_large, log_time_large, 1)
                
                problem_size_large = df_large['n'].values * df_large['m'].values
                log_size_large = np.log(problem_size_large)
                coeffs_nm_large = np.polyfit(log_size_large, log_time_large, 1)
            else:
                coeffs_n_large = [0, 0]
                coeffs_nm_large = [0, 0]
            
            print(f"Empirical scaling (all data): time ∝ n^{coeffs_n[0]:.2f}")
            print(f"Empirical scaling (n×m): time ∝ (n×m)^{coeffs_nm[0]:.2f}")
            
            if coeffs_n_large[0] > 0:
                print(f"\nLarge instances (n > 1000):")
                print(f"  Empirical: time ∝ n^{coeffs_n_large[0]:.2f}")
            
            
            
        
    
    def create_scalability_plots(self, df):
        """Create scalability plots as in the paper"""
        os.makedirs(self.run_dir, exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Scalability Analysis - Real Data Austria', fontsize=16)
        
        # 1. Time vs Size (by city)
        ax1 = axes[0, 0]
        for city in df['city'].unique():
            city_data = df[df['city'] == city].sort_values('n')
            ax1.loglog(city_data['n'], city_data['total_time'], 
                      'o-', label=city, markersize=8, linewidth=2)
        
        ax1.set_xlabel('Number of demand points (n)', fontsize=12)
        ax1.set_ylabel('Total time (seconds)', fontsize=12)
        ax1.set_title('Time Scalability by City', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3, which="both")
        
        # 2. Build vs Solve time
        ax2 = axes[0, 1]
        ax2.scatter(df['n'], df['build_time'], alpha=0.6, label='Build time', s=50)
        ax2.scatter(df['n'], df['solve_time'], alpha=0.6, label='Solve time', s=50)
        ax2.set_xlabel('Number of demand points (n)', fontsize=12)
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.set_title('Build Time vs Solve Time', fontsize=14)
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Solution cost vs size
        ax3 = axes[1, 0]
        for city in df['city'].unique():
            city_data = df[df['city'] == city].sort_values('n')
            ax3.plot(city_data['n'], city_data['objective']/1000, 
                    'o-', label=city, markersize=8, linewidth=2)
        
        ax3.set_xlabel('Number of demand points (n)', fontsize=12)
        ax3.set_ylabel('Solution cost (k$)', fontsize=12)
        ax3.set_title('Optimal Cost vs Size', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Open SPs vs size
        ax4 = axes[1, 1]
        for city in df['city'].unique():
            city_data = df[df['city'] == city].sort_values('n')
            ax4.plot(city_data['n'], city_data['n_sps_opened'], 
                    'o-', label=city, markersize=8, linewidth=2)
        
        ax4.set_xlabel('Number of demand points (n)', fontsize=12)
        ax4.set_ylabel('Number of open SPs', fontsize=12)
        ax4.set_title('Service Points Opened vs Size', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_path = os.path.join(self.run_dir, 'scalability_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: {plot_path}")
        plt.close()
        
        # Create simple complexity analysis plot (as shown in the image)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Data
        n_values = df['n'].values
        time_values = df['total_time'].values
        
        # Plot real data points
        ax.loglog(n_values, time_values, 'bo', markersize=10, label='Real data', alpha=0.9)
        
        # Fit all data
        log_n = np.log(n_values)
        log_time = np.log(time_values)
        coeffs = np.polyfit(log_n, log_time, 1)
        
        # Create fit line
        n_fit = np.logspace(np.log10(n_values.min()*0.8), np.log10(n_values.max()*1.2), 100)
        time_fit = np.exp(coeffs[1]) * n_fit**coeffs[0]
        ax.loglog(n_fit, time_fit, 'r--', linewidth=2.5, label='Fit', alpha=0.8)
        
        # Labels and title
        ax.set_xlabel('Number of demand points (n)', fontsize=14)
        ax.set_ylabel('Total time (seconds)', fontsize=14)
        ax.set_title('Computational Complexity Analysis', fontsize=16, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(fontsize=12, loc='upper left')
        
        # Grid
        ax.grid(True, alpha=0.3, which="both", linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.15, which="minor", linestyle=':', linewidth=0.3)
        
        # Set axis limits to match the image
        ax.set_xlim(300, 20000)
        ax.set_ylim(5, 3000)
        
        # Adjust tick labels
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        
        # Save
        plot_path2 = os.path.join(self.run_dir, 'complexity_analysis.png')
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"Complexity analysis plot saved to: {plot_path2}")
        plt.close()
    
    def save_results(self):
        """Save the results"""
        os.makedirs(self.run_dir, exist_ok=True)

        # Detailed JSON
        with open(os.path.join(self.run_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # CSV for analysis
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(os.path.join(self.run_dir, 'results.csv'), index=False)
    
    def generate_report(self):
        """Generate text format report"""
        
        report_path = os.path.join(self.run_dir, 'report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SCALABILITY TEST REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Model: SP Location and Capacity (Raviv, 2023)\n")
            f.write("Data: OpenStreetMap Austria (Vienna, Graz, Linz)\n\n")
            
            f.write("TEST CONFIGURATION:\n")
            f.write("-"*50 + "\n")
            f.write("- Pickup probability (p): 0.5\n")
            f.write("- Rejection cost (α): 10\n")
            f.write("- Service radius: 300m\n")
            f.write("- Capacities: S = {30, 60, 90}\n")
            f.write("- Setup costs: base=10k$, var=1/6 * capacity\n\n")
            
            if self.results:
                df = pd.DataFrame(self.results)
                
                f.write("RESULTS:\n")
                f.write("-"*50 + "\n")
                f.write(f"Instances tested: {len(df)}\n")
                f.write(f"Range n: {df['n'].min()} - {df['n'].max()}\n")
                f.write(f"Total test time: {df['total_time'].sum():.1f}s\n\n")
                
                # Results by city
                for city in sorted(df['city'].unique()):
                    city_data = df[df['city'] == city]
                    f.write(f"\n{city.upper()}:\n")
                    f.write(f"  Instances: {len(city_data)}\n")
                    f.write(f"  Demand points: {city_data['n'].min()} - {city_data['n'].max()}\n")
                    f.write(f"  Average time: {city_data['total_time'].mean():.2f}s\n")
                    f.write(f"  Average cost: ${city_data['objective'].mean():,.0f}\n")
                    f.write(f"  Average SPs: {city_data['n_sps_opened'].mean():.1f}\n")
                
                # Complexity
                if len(df) >= 5:
                    log_n = np.log(df['n'].values)
                    log_time = np.log(df['total_time'].values)
                    coeffs = np.polyfit(log_n, log_time, 1)
                    
                    f.write(f"\n\nCOMPUTATIONAL COMPLEXITY:\n")
                    f.write("-"*50 + "\n")
                    f.write(f"Estimated from data: O(n^{coeffs[0]:.2f})\n")
                    f.write(f"Paper (theoretical): O(n^3)\n")
                    f.write(f"Paper (practical): O(n^2)\n")
        
        print(f"\nReport saved to: {report_path}")


def main():
    """Main function"""
    
    print("SCALABILITY TEST - Service Points Location and Capacity Problem")
    print("Based on: Tal Raviv (2023) - Transportation Research Part E")
    print("Real data: Vienna, Graz, Linz (OpenStreetMap)")
    
    # Create tester
    tester = ScalabilityTester()
    
    try:
        # Run test
        tester.run_scalability_test()
        
        # Generate report
        tester.generate_report()
        
        print("\n" + "="*70)
        print("SCALABILITY TEST COMPLETED!")
        print(f"Results saved in: {tester.run_dir}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user!")
        tester.save_results()
        tester.generate_report()
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        tester.save_results()


if __name__ == "__main__":
    main()