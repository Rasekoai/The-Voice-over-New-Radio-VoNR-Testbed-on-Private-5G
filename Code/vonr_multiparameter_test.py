#!/usr/bin/env python3

import subprocess
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Configuration
OUTPUT_DIR = "/home/bluelabuser/vonr_multiparameter_test"
MODEL_DIR = "/home/bluelabuser/model"
TEST_DURATION = 30
CAPTURE_INTERFACE = "ogstun"
UE2_IP = "10.46.0.10"

# SUDO PASSWORD
SUDO_PASSWORD = "blu3l0ck"

# Single impairment tests
# Single impairment tests - REDUCED VALUES
SINGLE_TESTS = {
    "loss": [0, 1, 2, 3, 5],           # Was [0, 5, 10, 15, 20]
    "jitter": [0, 10, 20, 30, 50],     # Was [0, 25, 50, 75, 100]
    "latency": [0, 25, 50, 75, 100]    # Was [0, 50, 100, 150, 200]
}

# Combined tests - LIGHTER VALUES
COMBINED_TESTS = [
    (1, 10, 25, "Light combined"),      # Was (2, 20, 50)
    (2, 15, 50, "Moderate combined"),   # Was (5, 30, 75)
    (3, 20, 75, "Heavy combined"),      # Was (10, 40, 100)
    (5, 30, 100, "Severe combined"),    # Was (15, 50, 150)
    (2, 0, 50, "Loss + Latency"),       # Was (5, 0, 100)
    (0, 20, 50, "Jitter + Latency"),    # Was (0, 50, 100)
    (2, 20, 0, "Loss + Jitter"),        # Was (5, 50, 0)
]

# Matrix - SMALLER RANGE
MATRIX_TESTS = {
    "loss": [0, 2, 5],          # Was [0, 5, 10]
    "jitter": [0, 10, 25],      # Was [0, 25, 50]
    "latency": [0, 25, 50]      # Was [0, 50, 100]
}
# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Calculate totals
single_total = sum(len(v) for v in SINGLE_TESTS.values())
combined_total = len(COMBINED_TESTS)
matrix_total = len(MATRIX_TESTS['loss']) * len(MATRIX_TESTS['jitter']) * len(MATRIX_TESTS['latency'])
# Subtract 1 from matrix because baseline (0,0,0) is already in single tests
matrix_total_unique = matrix_total - 1

total_tests = single_total + combined_total + matrix_total_unique
total_time = total_tests * TEST_DURATION / 60

print("="*70)
print("VoNR COMPREHENSIVE MULTI-PARAMETER QUALITY TEST")
print("="*70)
print()
print(f"Configuration:")
print(f"  UE2 (WiFi) IP: {UE2_IP}")
print(f"  Test Duration: {TEST_DURATION}s per sample")
print(f"  Capture Interface: {CAPTURE_INTERFACE}")
print(f"  Output: {OUTPUT_DIR}/")
print()
print(f"Test Plan:")
print(f"  PHASE 1 - Single Parameters: {single_total} tests")
print(f"     Packet Loss: {SINGLE_TESTS['loss']}")
print(f"     Jitter: {SINGLE_TESTS['jitter']}")
print(f"     Latency: {SINGLE_TESTS['latency']}")
print()
print(f"  PHASE 2 - Combined Parameters: {combined_total} tests")
for loss, jitter, latency, desc in COMBINED_TESTS[:5]:
    print(f"    • {desc}: L={loss}%, J={jitter}ms, D={latency}ms")
if len(COMBINED_TESTS) > 5:
    print(f"    ... and {len(COMBINED_TESTS)-5} more")
print()
print(f"  PHASE 3 - Full Matrix: {matrix_total_unique} tests")
print(f"    Loss: {MATRIX_TESTS['loss']}")
print(f"    Jitter: {MATRIX_TESTS['jitter']}")
print(f"    Latency: {MATRIX_TESTS['latency']}")
print()
print(f"Grand Total: {total_tests} tests, ~{total_time:.1f} minutes")
print()
print("Prerequisites:")
print("  ✓ Active call running")
print("  ✓ Reference audio looping")
print("  ✓ Call must stay up for entire duration!")
print()
input("Press ENTER to start comprehensive multi-parameter test...")


def run_sudo_command(cmd, timeout=60):
    """Run a sudo command with password if needed"""
    if SUDO_PASSWORD:
        full_cmd = f"echo '{SUDO_PASSWORD}' | sudo -S {cmd}"
    else:
        full_cmd = f"sudo {cmd}"
    return subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def capture_rtp_traffic(duration, output_pcap):
    """Capture RTP traffic destined to UE2 with progress indicator"""
    #tcpdump_cmd = f"timeout {duration + 5} sudo tcpdump -i {CAPTURE_INTERFACE} -w {output_pcap} 'udp and portrange 10000-60000 and dst host {UE2_IP}' 2>/dev/null &"
    tcpdump_cmd = f"timeout {duration + 5} sudo tcpdump -i {CAPTURE_INTERFACE} -w {output_pcap} 'udp and portrange 10000-60000' &"   
    subprocess.run(tcpdump_cmd, shell=True)
    
    for i in range(duration):
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(".", end="", flush=True)
    
    time.sleep(2)
    subprocess.run("sudo pkill -f 'tcpdump.*ogstun'", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(1)
    
    if os.path.exists(output_pcap):
        size = os.path.getsize(output_pcap)
        if size > 10000:
            print(f" ✓ {size/1024:.0f}KB")
            return True
        else:
            print(f" ✗ Too small")
            return False
    else:
        print(" ✗ Failed")
        return False


def extract_audio_from_pcap(pcap_file, test_name):
    """Extract audio from PCAP file"""
    try:
        pcap_size = os.path.getsize(pcap_file)
        if pcap_size < 10000:
            return False
        
        port_cmd = f"tshark -r {pcap_file} -T fields -e udp.dstport 2>&1 | grep -v '^Running' | grep -v '^$' | sort -u | grep -E '^[0-9]+' | head -5"
        result = subprocess.run(port_cmd, shell=True, capture_output=True, text=True)
        ports = [p for p in result.stdout.strip().split('\n') if p and p.isdigit()]
        
        if not ports:
            return False
        
        decode_opts = " ".join([f"-d udp.port=={port},rtp" for port in ports])
        
        ssrc_cmd = f"tshark -r {pcap_file} {decode_opts} -T fields -e rtp.ssrc 2>/dev/null | sort -u | grep -v '^$' | head -1"
        result = subprocess.run(ssrc_cmd, shell=True, capture_output=True, text=True)
        ssrc = result.stdout.strip()
        
        if not ssrc:
            return False
        
        #payload_cmd = f"tshark -r {pcap_file} {decode_opts} -Y 'rtp.ssrc=={ssrc}' -T fields -e rtp.payload 2>/dev/null | xxd -r -p > /tmp/stream.raw"
        payload_cmd = f"tshark -r {pcap_file} {decode_opts} -Y 'rtp.ssrc=={ssrc}' -T fields -e rtp.payload 2>/dev/null | xxd -r -p | sudo tee /tmp/stream.raw > /dev/null"
        subprocess.run(payload_cmd, shell=True, timeout=60)
        
        if not os.path.exists('/tmp/stream.raw') or os.path.getsize('/tmp/stream.raw') == 0:
            subprocess.run("rm /tmp/stream.raw 2>/dev/null", shell=True)
            return False
        
        target_size = 30 * 8000
        raw_size = os.path.getsize('/tmp/stream.raw')
        if raw_size > target_size:
            subprocess.run(f"tail -c {target_size} /tmp/stream.raw > /tmp/stream_tail.raw", shell=True)
            subprocess.run("mv /tmp/stream_tail.raw /tmp/stream.raw", shell=True)
        
        output_file = f"{OUTPUT_DIR}/{test_name}.wav"
        ffmpeg_cmd = f"ffmpeg -f mulaw -ar 8000 -ac 1 -i /tmp/stream.raw {output_file} -y"
        subprocess.run(ffmpeg_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        subprocess.run("rm /tmp/stream.raw 2>/dev/null", shell=True)
        
        return os.path.exists(output_file)
        
    except Exception as e:
        subprocess.run("rm /tmp/stream.raw 2>/dev/null", shell=True)
        return False


def calculate_mos(ref_file, deg_file):
    """Calculate MOS using ViSQOL"""
    try:
        tflite_model = f"{MODEL_DIR}/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        
        cmd = f"visqol --reference_file {ref_file} --degraded_file {deg_file}"
        cmd += f" --use_speech_mode --similarity_to_quality_model {tflite_model}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        for line in result.stdout.split('\n'):
            if "MOS-LQO:" in line:
                return float(line.split()[1])
        
        return None
    except Exception as e:
        return None


def apply_impairments(loss=0, jitter=0, latency=0):
    """Apply multiple network impairments simultaneously"""
    run_sudo_command("tc qdisc del dev ogstun root 2>/dev/null")
    time.sleep(1)
    
    if loss == 0 and jitter == 0 and latency == 0:
        return "baseline"
    
    # Build tc netem command with all impairments
    cmd_parts = ["tc qdisc add dev ogstun root netem"]
    
    if latency > 0:
        if jitter > 0:
            cmd_parts.append(f"delay {latency}ms {jitter}ms distribution normal")
        else:
            cmd_parts.append(f"delay {latency}ms")
    elif jitter > 0:
        # If jitter without base latency, add small base delay
        cmd_parts.append(f"delay 10ms {jitter}ms distribution normal")
    
    if loss > 0:
        cmd_parts.append(f"loss {loss}%")
    
    cmd = " ".join(cmd_parts)
    run_sudo_command(cmd)
    
    return f"L{loss}_J{jitter}_D{latency}"


def run_single_test(loss, jitter, latency, label, test_num, total, reference_file):
    """Run a single test with specified impairments"""
    
    print(f"[{test_num}/{total}] {label}...", end=" ", flush=True)
    
    # Apply impairments
    impairment_name = apply_impairments(loss, jitter, latency)
    
    # Capture
    pcap_file = f"/tmp/capture_{impairment_name}.pcap"
    if not capture_rtp_traffic(TEST_DURATION, pcap_file):
        print(f"      Capture failed")
        return {
            "loss": loss, "jitter": jitter, "latency": latency,
            "label": label, "mos": None, "status": "capture_failed",
            "test_type": "unknown"
        }
    
    # Extract audio
    audio_file = f"test_{impairment_name}"
    if not extract_audio_from_pcap(pcap_file, audio_file):
        print(f"     Audio extraction failed")
        os.remove(pcap_file)
        return {
            "loss": loss, "jitter": jitter, "latency": latency,
            "label": label, "mos": None, "status": "extraction_failed",
            "test_type": "unknown"
        }
    
    os.remove(pcap_file)
    
    # Calculate MOS
    degraded_file = f"{OUTPUT_DIR}/{audio_file}.wav"
    
    if loss == 0 and jitter == 0 and latency == 0:
        # This is the baseline
        mos = calculate_mos(degraded_file, degraded_file)
        mos_str = f"{mos:.3f}" if mos else "N/A"
        print(f"    Reference. MOS: {mos_str}")
        return {
            "loss": loss, "jitter": jitter, "latency": latency,
            "label": label, "mos": mos, "status": "success",
            "reference_file": degraded_file, "test_type": "baseline"
        }
    else:
        if reference_file:
            mos = calculate_mos(reference_file, degraded_file)
            mos_str = f"{mos:.3f}" if mos else "N/A"
            print(f"    MOS: {mos_str}")
            return {
                "loss": loss, "jitter": jitter, "latency": latency,
                "label": label, "mos": mos, 
                "status": "success" if mos else "mos_failed",
                "test_type": "unknown"
            }
        else:
            print(f"     No reference")
            return {
                "loss": loss, "jitter": jitter, "latency": latency,
                "label": label, "mos": None, "status": "no_reference",
                "test_type": "unknown"
            }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

all_results = []
start_time = time.time()
reference_file = None
test_counter = 1

# PHASE 1: Single parameter tests
print(f"\n{'='*70}")
print(f"PHASE 1: SINGLE PARAMETER TESTING")
print(f"{'='*70}\n")

for imp_type, values in SINGLE_TESTS.items():
    print(f"\nTesting {imp_type.upper()}:")
    print("-" * 70)
    
    for value in values:
        if imp_type == "loss":
            loss, jitter, latency = value, 0, 0
        elif imp_type == "jitter":
            loss, jitter, latency = 0, value, 0
        else:
            loss, jitter, latency = 0, 0, value
        
        unit = "%" if imp_type == "loss" else "ms"
        label = "Baseline" if value == 0 else f"{imp_type.title()} {value}{unit}"
        
        result = run_single_test(loss, jitter, latency, label, 
                                test_counter, total_tests, reference_file)
        
        result['test_type'] = 'baseline' if value == 0 else 'single'
        
        if value == 0 and 'reference_file' in result:
            reference_file = result['reference_file']
        
        all_results.append(result)
        test_counter += 1

print(f"\n✓ Phase 1 complete. Taking 5s break...\n")
time.sleep(5)

# PHASE 2: Combined parameter tests
print(f"\n{'='*70}")
print(f"PHASE 2: COMBINED PARAMETER TESTING")
print(f"{'='*70}\n")

for loss, jitter, latency, desc in COMBINED_TESTS:
    result = run_single_test(loss, jitter, latency, desc, 
                            test_counter, total_tests, reference_file)
    result['test_type'] = 'combined'
    all_results.append(result)
    test_counter += 1

print(f"\n✓ Phase 2 complete. Taking 5s break...\n")
time.sleep(5)

# PHASE 3: Full matrix tests
print(f"\n{'='*70}")
print(f"PHASE 3: FULL MATRIX TESTING")
print(f"{'='*70}\n")

for loss in MATRIX_TESTS['loss']:
    for jitter in MATRIX_TESTS['jitter']:
        for latency in MATRIX_TESTS['latency']:
            # Skip baseline as it was already done in Phase 1
            if loss == 0 and jitter == 0 and latency == 0:
                continue
            
            # Skip if already tested in single parameter phase
            already_tested = False
            if (loss > 0 and jitter == 0 and latency == 0) or \
               (loss == 0 and jitter > 0 and latency == 0) or \
               (loss == 0 and jitter == 0 and latency > 0):
                if (loss in SINGLE_TESTS.get('loss', []) and jitter == 0 and latency == 0) or \
                   (jitter in SINGLE_TESTS.get('jitter', []) and loss == 0 and latency == 0) or \
                   (latency in SINGLE_TESTS.get('latency', []) and loss == 0 and jitter == 0):
                    already_tested = True
            
            if already_tested:
                continue
            
            label = f"L{loss}% J{jitter}ms D{latency}ms"
            
            result = run_single_test(loss, jitter, latency, label, 
                                    test_counter, total_tests, reference_file)
            result['test_type'] = 'matrix'
            all_results.append(result)
            test_counter += 1

# Remove impairments
run_sudo_command("tc qdisc del dev ogstun root 2>/dev/null")

elapsed = time.time() - start_time

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

df = pd.DataFrame(all_results)
csv_path = f"{OUTPUT_DIR}/results_{timestamp}.csv"
df.to_csv(csv_path, index=False)
print(f"✓ CSV saved: {csv_path}")

# =============================================================================
# GENERATE GRAPHS
# =============================================================================

print("\nGenerating graphs...")

success_df = df[df['status'] == 'success']

# GRAPH 1: Single parameter impacts (3-panel)
print("  Creating Graph 1: Single parameter impacts...")
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, imp_type in enumerate(["loss", "jitter", "latency"]):
    ax = axes[idx]
    data = success_df[success_df['test_type'].isin(['baseline', 'single'])].copy()
    
    # Filter for this parameter only
    other_params = [p for p in ['loss', 'jitter', 'latency'] if p != imp_type]
    data = data[(data[other_params[0]] == 0) & (data[other_params[1]] == 0)]
    data = data.sort_values(imp_type)
    
    if len(data) > 0:
        x = data[imp_type]
        y = data['mos']
        ax.plot(x, y, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax.fill_between(x, y, 1, alpha=0.2, color='#2E86AB')
        
        ax.set_xlabel(f"{imp_type.title()} ({'%' if imp_type == 'loss' else 'ms'})", 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('MOS Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Impact of {imp_type.title()}', fontsize=14, fontweight='bold')
        ax.set_ylim(1, 5)
        ax.grid(True, alpha=0.3)
        
        # Quality zones
        ax.axhspan(4.5, 5, alpha=0.1, color='green')
        ax.axhspan(4.0, 4.5, alpha=0.1, color='lightgreen')
        ax.axhspan(3.5, 4.0, alpha=0.1, color='yellow')
        ax.axhspan(3.0, 3.5, alpha=0.1, color='orange')
        ax.axhspan(1, 3.0, alpha=0.1, color='red')
        
        # Value labels
        for xi, yi in zip(x, y):
            if yi and not pd.isna(yi):
                ax.annotate(f'{yi:.2f}', (xi, yi), textcoords="offset points", 
                           xytext=(0,8), ha='center', fontsize=8)

plt.suptitle(f'Single Parameter Impact on VoNR Quality', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
graph1_path = f"{OUTPUT_DIR}/graph1_single_params_{timestamp}.png"
plt.savefig(graph1_path, dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: {graph1_path}")
plt.close()

# GRAPH 2: Combined parameter tests (horizontal bar chart)
print("  Creating Graph 2: Combined parameter tests...")
fig2, ax = plt.subplots(figsize=(14, 8))

combined_data = success_df[success_df['test_type'] == 'combined'].copy()
if len(combined_data) > 0:
    combined_data = combined_data.sort_values('mos', ascending=True)
    
    colors = plt.cm.RdYlGn(combined_data['mos'] / 5.0)
    
    bars = ax.barh(range(len(combined_data)), combined_data['mos'], color=colors)
    ax.set_yticks(range(len(combined_data)))
    ax.set_yticklabels(combined_data['label'], fontsize=10)
    ax.set_xlabel('MOS Score', fontsize=12, fontweight='bold')
    ax.set_title('Combined Impairments - MOS Scores', fontsize=14, fontweight='bold')
    ax.set_xlim(1, 5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(combined_data.iterrows()):
        ax.text(row['mos'] + 0.05, i, f"{row['mos']:.2f}", 
               va='center', fontsize=9, fontweight='bold')
    
    # Add parameter labels
    for i, (idx, row) in enumerate(combined_data.iterrows()):
        param_text = f"L:{row['loss']}% J:{row['jitter']}ms D:{row['latency']}ms"
        ax.text(0.05, i, param_text, va='center', fontsize=7, 
               color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

plt.tight_layout()
graph2_path = f"{OUTPUT_DIR}/graph2_combined_{timestamp}.png"
plt.savefig(graph2_path, dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: {graph2_path}")
plt.close()

# GRAPH 3: Full matrix heatmaps (3-panel showing interactions)
print("  Creating Graph 3: Full matrix heatmaps...")
fig3, axes = plt.subplots(1, 3, figsize=(20, 6))

matrix_data = success_df[success_df['test_type'].isin(['baseline', 'single', 'matrix'])].copy()

pairs = [
    ('loss', 'jitter', 'latency', 0, '%', 'ms'),
    ('loss', 'latency', 'jitter', 1, '%', 'ms'),
    ('jitter', 'latency', 'loss', 2, 'ms', 'ms')
]

for x_param, y_param, fixed_param, idx, x_unit, y_unit in pairs:
    ax = axes[idx]
    
    # Use middle value of fixed parameter
    fixed_vals = sorted(matrix_data[fixed_param].unique())
    if len(fixed_vals) > 0:
        fixed_val = fixed_vals[len(fixed_vals)//2]
        
        subset = matrix_data[matrix_data[fixed_param] == fixed_val]
        
        if len(subset) > 1:
            pivot = subset.pivot_table(values='mos', index=y_param, columns=x_param, aggfunc='mean')
            
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                       vmin=1, vmax=5, ax=ax, cbar_kws={'label': 'MOS Score'})
            
            ax.set_xlabel(f"{x_param.title()} ({x_unit})", fontsize=11, fontweight='bold')
            ax.set_ylabel(f"{y_param.title()} ({y_unit})", fontsize=11, fontweight='bold')
            
            fixed_unit = '%' if fixed_param == 'loss' else 'ms'
            ax.set_title(f'{x_param.title()} vs {y_param.title()}\n(Fixed: {fixed_param.title()}={fixed_val}{fixed_unit})', 
                        fontsize=12, fontweight='bold')

plt.suptitle(f'Multi-Parameter Interaction Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
graph3_path = f"{OUTPUT_DIR}/graph3_matrix_{timestamp}.png"
plt.savefig(graph3_path, dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: {graph3_path}")
plt.close()

print("✓ All graphs generated")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
print(f"\nTotal time: {elapsed/60:.1f} minutes")
print(f"Tests completed: {len(success_df)}/{len(df)}")
print(f"\nOutput directory: {OUTPUT_DIR}/")
print(f"  - results_{timestamp}.csv")
print(f"  - graph1_single_params_{timestamp}.png")
print(f"  - graph2_combined_{timestamp}.png")
print(f"  - graph3_matrix_{timestamp}.png")
print(f"  - {len(df)} WAV files")

# Show statistics by phase
print("\n" + "-"*70)
print("RESULTS BY PHASE:")
print("-" * 70)

for phase_type, phase_name in [('single', 'Single Parameters'), 
                                ('combined', 'Combined Tests'), 
                                ('matrix', 'Matrix Tests')]:
    phase_data = success_df[success_df['test_type'] == phase_type]
    if len(phase_data) > 0:
        avg_mos = phase_data['mos'].mean()
        min_mos = phase_data['mos'].min()
        max_mos = phase_data['mos'].max()
        print(f"{phase_name:20s}: {len(phase_data):2d} tests | Avg MOS: {avg_mos:.3f} | Range: {min_mos:.3f} - {max_mos:.3f}")

# Show top/bottom results overall
print("\n" + "-"*70)
print("BEST QUALITY (Top 5):")
for idx, row in success_df.nlargest(5, 'mos').iterrows():
    print(f"  {row['label']:35s} MOS: {row['mos']:.3f} | L:{row['loss']:2.0f}% J:{row['jitter']:3.0f}ms D:{row['latency']:3.0f}ms")

print("\nWORST QUALITY (Bottom 5):")
for idx, row in success_df.nsmallest(5, 'mos').iterrows():
    print(f"  {row['label']:35s} MOS: {row['mos']:.3f} | L:{row['loss']:2.0f}% J:{row['jitter']:3.0f}ms D:{row['latency']:3.0f}ms")

print("="*70)
