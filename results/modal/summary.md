| job | method | ranks | PPL (all) | PPL (post-freeze) | mem | s/window |
|---|---|---|---|---|---|---|
| mistral7b_pg19 | exact | - | 5.728 | 5.732 | 1.000 | 0.44 |
| mistral7b_pg19 | tkv_sink | r=64 | 5.871 | 5.878 | 0.594 | 4.26 |
| mistral7b_pg19 | streaming_llm | r=64 | 5.765 | 5.770 | 0.594 | 1.60 |
| mistral7b_shift | exact | - | 2.646 | 2.314 | 1.000 | 0.46 |
| mistral7b_shift | tkv_sink | r=16 | 3.073 | 2.745 | 0.207 | 4.27 |
| mistral7b_shift | tkv_sink | r=64 | 2.753 | 2.421 | 0.594 | 4.28 |
| mistral7b_shift | streaming_llm | r=16 | 2.747 | 2.414 | 0.207 | 1.61 |
| mistral7b_shift | streaming_llm | r=64 | 2.651 | 2.318 | 0.594 | 1.60 |
| mistral7b_shift | oja_sink | r=64 | 2.744 | 2.412 | 0.594 | 40.71 |
| mistral7b_wikitext2 | exact | - | 4.605 | 4.595 | 1.000 | 0.45 |
| mistral7b_wikitext2 | tkv_sink | r=16 | 5.502 | 5.505 | 0.207 | 4.28 |
| mistral7b_wikitext2 | tkv_sink | r=32 | 5.171 | 5.168 | 0.336 | 4.28 |
| mistral7b_wikitext2 | tkv_sink | r=64 | 4.860 | 4.853 | 0.594 | 4.28 |
| mistral7b_wikitext2 | tkv_sink | pyr[96, 64, 48, 32] | 4.967 | 4.962 | 0.562 | 4.27 |
| mistral7b_wikitext2 | streaming_llm | r=16 | 5.035 | 5.031 | 0.207 | 1.59 |
| mistral7b_wikitext2 | streaming_llm | r=32 | 4.821 | 4.813 | 0.336 | 1.59 |
| mistral7b_wikitext2 | streaming_llm | r=64 | 4.650 | 4.640 | 0.594 | 1.59 |
| mistral7b_wikitext2 | streaming_llm | pyr[96, 64, 48, 32] | 4.732 | 4.724 | 0.562 | 1.59 |
| mistral7b_wikitext2 | oja_sink | r=64 | 4.842 | 4.835 | 0.594 | 41.09 |
| mistral7b_wikitext2 | tkv_sink | r=64 | 5.127 | 5.124 | 0.594 | 4.28 |
| qwen14b_pg19 | exact | - | 6.323 | 6.330 | 1.000 | 0.70 |
| qwen14b_pg19 | tkv_sink | r=64 | 6.468 | 6.478 | 0.594 | 8.23 |
| qwen14b_pg19 | streaming_llm | r=64 | 6.376 | 6.384 | 0.594 | 2.49 |
| qwen14b_shift | exact | - | 2.540 | 2.311 | 1.000 | 0.64 |
| qwen14b_shift | tkv_sink | r=16 | 2.946 | 2.739 | 0.207 | 8.09 |
| qwen14b_shift | tkv_sink | r=64 | 2.661 | 2.438 | 0.594 | 8.08 |
| qwen14b_shift | streaming_llm | r=16 | 2.650 | 2.426 | 0.207 | 2.34 |
| qwen14b_shift | streaming_llm | r=64 | 2.543 | 2.315 | 0.594 | 2.34 |
| qwen14b_shift | oja_sink | r=64 | 2.649 | 2.425 | 0.594 | 91.24 |
| qwen14b_wikitext2 | exact | - | 4.004 | 3.977 | 1.000 | 0.63 |
| qwen14b_wikitext2 | tkv_sink | r=16 | 4.907 | 4.890 | 0.207 | 8.07 |
| qwen14b_wikitext2 | tkv_sink | r=32 | 4.544 | 4.522 | 0.336 | 8.07 |
| qwen14b_wikitext2 | tkv_sink | r=64 | 4.214 | 4.189 | 0.594 | 8.08 |
| qwen14b_wikitext2 | tkv_sink | pyr[96, 64, 48, 32] | 4.362 | 4.339 | 0.562 | 8.06 |
| qwen14b_wikitext2 | streaming_llm | r=16 | 4.293 | 4.270 | 0.207 | 2.32 |
| qwen14b_wikitext2 | streaming_llm | r=32 | 4.140 | 4.115 | 0.336 | 2.31 |
| qwen14b_wikitext2 | streaming_llm | r=64 | 4.044 | 4.018 | 0.594 | 2.31 |
| qwen14b_wikitext2 | streaming_llm | pyr[96, 64, 48, 32] | 4.104 | 4.079 | 0.562 | 2.31 |
| qwen14b_wikitext2 | oja_sink | r=64 | 4.205 | 4.180 | 0.594 | 91.00 |
| qwen14b_wikitext2 | tkv_sink | r=64 | 4.400 | 4.377 | 0.594 | 8.07 |
