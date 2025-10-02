#!/bin/bash

# Script to run AutoInterp with custom financial prompt
# This script temporarily replaces the default prompt with a financial-specific one

echo "🎯 AutoInterp with Custom Financial Prompt"
echo "=========================================="
echo "This script will temporarily replace the default prompt with a financial-specific one"
echo "for more specific and less generic feature explanations."
echo ""

# Check if we're in the right directory
if [ ! -f "run_multi_layer_top10_analysis_70b_financial.sh" ]; then
    echo "❌ Please run this script from the FinanceLabeling directory"
    exit 1
fi

# Backup the original prompt file
echo "📋 Backing up original prompt file..."
cp /home/nvidia/Documents/Hariom/autointerp/autointerp_full/autointerp_full/explainers/default/prompts.py /home/nvidia/Documents/Hariom/autointerp/autointerp_full/autointerp_full/explainers/default/prompts.py.backup

# Replace the SYSTEM_CONTRASTIVE with the financial one
echo "🔄 Replacing default prompt with financial-specific prompt..."
python3 -c "
import re

# Read the financial prompt
with open('/home/nvidia/Documents/Hariom/autointerp/autointerp_full/autointerp_full/explainers/default/prompts_financial.py', 'r') as f:
    financial_content = f.read()

# Extract the SYSTEM_CONTRASTIVE from financial prompt
financial_match = re.search(r'SYSTEM_CONTRASTIVE = \"\"\"(.*?)\"\"\"', financial_content, re.DOTALL)
if financial_match:
    financial_system = financial_match.group(1)
    
    # Read the original prompts.py
    with open('/home/nvidia/Documents/Hariom/autointerp/autointerp_full/autointerp_full/explainers/default/prompts.py', 'r') as f:
        original_content = f.read()
    
    # Replace the SYSTEM_CONTRASTIVE in the original file
    new_content = re.sub(
        r'SYSTEM_CONTRASTIVE = \"\"\"(.*?)\"\"\"',
        f'SYSTEM_CONTRASTIVE = \"\"\"{financial_system}\"\"\"',
        original_content,
        flags=re.DOTALL
    )
    
    # Write the modified content
    with open('/home/nvidia/Documents/Hariom/autointerp/autointerp_full/autointerp_full/explainers/default/prompts.py', 'w') as f:
        f.write(new_content)
    
    print('✅ Successfully replaced SYSTEM_CONTRASTIVE with financial prompt')
    print('📝 Financial prompt focuses on specific financial concepts instead of generic terms')
else:
    print('❌ Could not extract financial prompt')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Failed to replace prompt. Restoring backup..."
    cp /home/nvidia/Documents/Hariom/autointerp/autointerp_full/autointerp_full/explainers/default/prompts.py.backup /home/nvidia/Documents/Hariom/autointerp/autointerp_full/autointerp_full/explainers/default/prompts.py
    exit 1
fi

echo ""
echo "🚀 Running AutoInterp with custom financial prompt..."
echo "This should produce more specific and less generic explanations."
echo ""

# Run the original script
./run_multi_layer_top10_analysis_70b_financial.sh

# Restore the original prompt file
echo ""
echo "🔄 Restoring original prompt file..."
cp /home/nvidia/Documents/Hariom/autointerp/autointerp_full/autointerp_full/explainers/default/prompts.py.backup /home/nvidia/Documents/Hariom/autointerp/autointerp_full/autointerp_full/explainers/default/prompts.py

echo "✅ Original prompt file restored"
echo ""
echo "🎯 Analysis complete with custom financial prompt!"
echo "Check the results to see if the explanations are more specific and less generic."
