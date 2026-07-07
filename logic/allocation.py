def build_allocation(scored_options, top_n=3):
    selected = scored_options[:top_n]
    total_score = sum(score for _, score in selected) or 1

    allocation = []
    for opt, score in selected:
        pct = round((score / total_score) * 100)
        allocation.append({"id": opt["id"], "name": opt["name"], "percentage": pct})

    # Rundungsfehler ausgleichen, damit Summe 100 ist
    diff = 100 - sum(x["percentage"] for x in allocation)
    if allocation:
        allocation[0]["percentage"] += diff

    return allocation
