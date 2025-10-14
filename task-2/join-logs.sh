grep -h . logs/log_rank_*.log | sort -k1,1 > logs/combined.log
rm logs/log_rank_*.log