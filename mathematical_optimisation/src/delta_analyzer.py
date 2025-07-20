import matplotlib.pyplot as plt

class DeltaAnalyzer:
    """Analyze the effect of delta parameter on the optimization"""
    
    @staticmethod
    def test_different_delta_values(optimizer, igp_util, verbose=False):
        """Test the effect of different delta values on the solution"""
        print("\n" + "="*60)
        print("TESTING DIFFERENT DELTA VALUES")
        print("="*60)
        
        # Test different delta values 
        delta_values = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        
        results = []
        
        print(f"\n{'Delta':>10} | {'Max Util %':>10} | {'LSPs':>6} | {'IGP %':>6} | {'MPLS %':>7} | {'Obj Value':>10}")
        print("-" * 70)
        
        for delta in delta_values:
            result = optimizer.solve_nominal(delta=delta, verbose=verbose)
            if result['status'] == 'optimal':
                results.append({
                    'delta': delta,
                    'result': result
                })
                print(f"{delta:10.0e} | {result['u_max_percent']:10.1f} | "
                      f"{result['lsp_count']:6d} | {result['igp_percent']:6.1f} | "
                      f"{result['mpls_percent']:7.1f} | {result['objective_value']:10.4f}")
        
        return results
    
    @staticmethod
    def plot_delta_analysis(delta_results, title="Effect of Delta on Solution"):
        """Plot the effect of delta on various metrics with improved layout"""
        if not delta_results:
            print("No results to plot")
            return
        
       
        fig = plt.figure(figsize=(16, 10))
        
        
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, 
                             top=0.92, bottom=0.08, left=0.08, right=0.95)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        deltas = [r['delta'] for r in delta_results]
        max_utils = [r['result']['u_max_percent'] for r in delta_results]
        lsp_counts = [r['result']['lsp_count'] for r in delta_results]
        mpls_percents = [r['result']['mpls_percent'] for r in delta_results]
        obj_values = [r['result']['objective_value'] for r in delta_results]
        
        # Plot 1: Max Utilization vs Delta
        ax1.semilogx(deltas, max_utils, 'o-', color='blue', linewidth=2, markersize=8)
        ax1.set_xlabel('Delta', fontsize=11)
        ax1.set_ylabel('Max Utilization (%)', fontsize=11)
        ax1.set_title('Maximum Utilization vs Delta', fontsize=12, pad=15)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=10)
        
        # Plot 2: Number of LSPs vs Delta
        ax2.semilogx(deltas, lsp_counts, 'o-', color='red', linewidth=2, markersize=8)
        ax2.set_xlabel('Delta', fontsize=11)
        ax2.set_ylabel('Number of LSPs', fontsize=11)
        ax2.set_title('Number of LSPs vs Delta', fontsize=12, pad=15)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=10)
        
        # Plot 3: MPLS Traffic % vs Delta
        ax3.semilogx(deltas, mpls_percents, 'o-', color='green', linewidth=2, markersize=8)
        ax3.set_xlabel('Delta', fontsize=11)
        ax3.set_ylabel('MPLS Traffic (%)', fontsize=11)
        ax3.set_title('MPLS Traffic Percentage vs Delta', fontsize=12, pad=15)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=10)
        
        # Plot 4: Objective Value vs Delta
        ax4.semilogx(deltas, obj_values, 'o-', color='purple', linewidth=2, markersize=8)
        ax4.set_xlabel('Delta', fontsize=11)
        ax4.set_ylabel('Objective Value', fontsize=11)
        ax4.set_title('Objective Function Value vs Delta', fontsize=12, pad=15)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', labelsize=10)
        
        
        fig.suptitle(title, fontsize=14, y=0.98)
        
        plt.savefig('delta_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()