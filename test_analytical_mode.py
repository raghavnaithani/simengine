#!/usr/bin/env python
"""Test longer simulation to see if we hit the fallback placeholder."""
import httpx
import json
import asyncio

async def test_analytical():
    """Test Analytical mode with more steps to see fallback behavior."""
    
    base_url = "http://localhost:8000"
    
    print("\n" + "="*80)
    print("TESTING ANALYTICAL MODE (MORE STEPS)")
    print("="*80 + "\n")
    
    payload = {
        "prompt": "A dental SaaS startup choosing between aggressive growth or conservative validation",
        "persona": "Skeptical Analyst",
        "mode": "Analytical",
        "simulate_steps": 3  # More steps to see branching
    }
    
    print(f"Starting Analytical simulation (3 steps)...")
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Start simulation
            response = await client.post(f"{base_url}/simulate/start", json=payload)
            response.raise_for_status()
            result = response.json()
            session_id = result.get("session_id")
            job_id = result.get("job_id")
            
            print(f"✓ Simulation started")
            print(f"  Session ID: {session_id}")
            print(f"  Job ID: {job_id}\n")
            
            # Poll for completion with time tracking
            max_polls = 300  # 5 minutes max
            poll_count = 0
            start_time = asyncio.get_event_loop().time()
            
            while poll_count < max_polls:
                await asyncio.sleep(2)
                poll_count += 1
                
                # Check job status
                status_response = await client.get(f"{base_url}/jobs/{job_id}")
                status_response.raise_for_status()
                status = status_response.json()
                
                if poll_count % 20 == 0:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    print(f"  Poll {poll_count} ({elapsed:.0f}s): {status.get('status')}")
                
                if status.get("status") == "completed":
                    elapsed = asyncio.get_event_loop().time() - start_time
                    print(f"\n✓ Simulation completed in {elapsed:.1f}s\n")
                    break
            
            # Get the graph/nodes
            graph_response = await client.get(f"{base_url}/graph?session_id={session_id}")
            graph_response.raise_for_status()
            graph = graph_response.json()
            
            nodes = graph.get("nodes", [])
            print(f"Total nodes generated: {len(nodes)}\n")
            
            # Check for fallback text
            fallback_text = "Generated with partial structured output"
            fallback_count = 0
            descriptions_by_step = {}
            
            print("="*80)
            print("NODE ANALYSIS")
            print("="*80 + "\n")
            
            for i, node in enumerate(nodes, 1):
                step = node.get('time_step', 0)
                if step not in descriptions_by_step:
                    descriptions_by_step[step] = []
                
                desc = node.get('description', '')
                descriptions_by_step[step].append(desc)
                
                if fallback_text in desc:
                    fallback_count += 1
                
                print(f"NODE {i} (Step {step}):")
                print(f"  Title: {node.get('title')}")
                print(f"  Description: {desc[:120]}...")
                if node.get('alternatives'):
                    print(f"  Alternatives: {len(node.get('alternatives'))}")
                print()
            
            # Diversity check
            print("="*80)
            print("DIVERSITY ANALYSIS")
            print("="*80)
            for step in sorted(descriptions_by_step.keys()):
                descs = descriptions_by_step[step]
                unique = len(set(descs))
                print(f"Step {step}: {len(descs)} nodes, {unique} unique descriptions")
                if unique == 1:
                    print(f"  ⚠ All identical: {descs[0][:80]}...")
                else:
                    for j, d in enumerate(set(descs), 1):
                        print(f"  Variant {j}: {d[:60]}...")
            
            if fallback_count > 0:
                print(f"\n⚠ FALLBACK TEXT DETECTED: {fallback_count}/{len(nodes)} nodes")
            else:
                print(f"\n✓ No fallback text detected")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_analytical())
