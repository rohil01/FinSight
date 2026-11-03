# Example reports

These files are produced by a live FinSight run. Reports are committed only after that run succeeds. The commands below write them here.

Regulatory — RBI payment aggregator guidelines and UPI-first companies:

```bash
finsight research "What are RBI's latest guidelines for payment aggregators, and how do they affect UPI-first companies?" --domain regulatory --max-iterations 3 -o examples/rbi-pa-guidelines.md
```

Market — UPI market share:

```bash
finsight research "How has UPI market share shifted among PhonePe, Google Pay, and Paytm?" --domain market --max-iterations 3 -o examples/upi-market-share.md
```
