#!/usr/bin/env Rscript
# Haplotype Network Visualization using pegas
# Generates THREE versions:
#   1. MST only (clean tree)
#   2. Filtered (MST + top N alternative edges by mutation count)
#   3. All edges (MST + all alternatives)
# Usage: Rscript haplotype_network.R <true_haplotypes.csv> <estimated_haplotypes.csv> <output_prefix> [num_alt_edges]
#   num_alt_edges: Optional. Number of alternative edges to show in filtered view (default: 50)
#                  Edges are selected by lowest mutation count first.

# Add user library path (for NixOS compatibility)
user_lib <- path.expand("~/R/library")
if (dir.exists(user_lib)) {
    .libPaths(c(user_lib, .libPaths()))
}

# Load required libraries
suppressPackageStartupMessages({
    library(ape)
    library(pegas)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
    cat("Usage: Rscript haplotype_network.R <true_haplotypes.csv> <estimated_haplotypes.csv> <output_prefix> [num_alt_edges]\n")
    cat("  num_alt_edges: Number of alternative edges to show in filtered view (default: 50)\n")
    cat("Generates: <output_prefix>_mst_only.png, <output_prefix>_filtered.png, <output_prefix>_all.png\n")
    quit(status = 1)
}

true_csv <- args[1]
est_csv <- args[2]
output_prefix <- args[3]

# Parse optional num_alt_edges argument (default: 50)
num_alt_edges <- 50
if (length(args) >= 4) {
    num_alt_edges <- as.integer(args[4])
    if (is.na(num_alt_edges) || num_alt_edges < 0) {
        cat("Warning: Invalid num_alt_edges value, using default (50)\n")
        num_alt_edges <- 50
    }
}

cat("Settings:\n")
cat("  Alternative edges to show:", num_alt_edges, "\n")

cat("Reading haplotype files...\n")

# Read CSV files
true_df <- read.csv(true_csv, stringsAsFactors = FALSE)
est_df <- read.csv(est_csv, stringsAsFactors = FALSE)

# Get sequences from first column
true_seqs <- as.character(true_df[, 1])
est_seqs <- as.character(est_df[, 1])

# Calculate total frequencies for each sequence
calculate_total_freq <- function(df) {
    seqs <- as.character(df[, 1])
    if (ncol(df) > 1) {
        freq_cols <- df[, 2:ncol(df), drop = FALSE]
        total_freqs <- rowSums(sapply(freq_cols, as.numeric), na.rm = TRUE)
    } else {
        total_freqs <- rep(1, nrow(df))
    }
    names(total_freqs) <- seqs
    return(total_freqs)
}

true_freqs <- calculate_total_freq(true_df)
est_freqs <- calculate_total_freq(est_df)

# Get unique sequences
true_seqs_unique <- unique(true_seqs)
est_seqs_unique <- unique(est_seqs)

# Categorize haplotypes
estimated_true <- intersect(true_seqs_unique, est_seqs_unique)
not_estimated_true <- setdiff(true_seqs_unique, est_seqs_unique)
estimated_not_true <- setdiff(est_seqs_unique, true_seqs_unique)

cat("\nHaplotype categories:\n")
cat("  Estimated True (dark blue):", length(estimated_true), "\n")
cat("  Not Estimated True (light blue):", length(not_estimated_true), "\n")
cat("  Estimated Not True (yellow):", length(estimated_not_true), "\n")

# Combine all haplotypes with their categories
all_seqs <- c(estimated_true, not_estimated_true, estimated_not_true)
categories <- c(
    rep("estimated_true", length(estimated_true)),
    rep("not_estimated_true", length(not_estimated_true)),
    rep("estimated_not_true", length(estimated_not_true))
)

# For true haplotypes, keep the original 0-based row index from true CSV
# (first occurrence if a sequence appears multiple times).
true_seq_csv_index <- match(all_seqs, true_seqs) - 1

if (length(all_seqs) == 0) {
    cat("Error: No haplotypes found!\n")
    quit(status = 1)
}

# Calculate combined frequency
combined_freqs <- sapply(all_seqs, function(seq) {
    freq <- 0
    if (seq %in% names(true_freqs)) {
        freq <- freq + sum(true_freqs[names(true_freqs) == seq])
    }
    if (seq %in% names(est_freqs)) {
        freq <- freq + sum(est_freqs[names(est_freqs) == seq])
    }
    return(freq)
})

cat("\nTotal unique haplotypes:", length(all_seqs), "\n")

# Create temporary FASTA file
temp_fasta <- tempfile(fileext = ".fasta")
fasta_lines <- character(0)
for (i in seq_along(all_seqs)) {
    seq <- gsub("-", "N", all_seqs[i])
    fasta_lines <- c(fasta_lines, paste0(">hap_", i), seq)
}
writeLines(fasta_lines, temp_fasta)

# Read sequences
cat("Reading sequences and creating haplotype network...\n")
seqs <- read.dna(temp_fasta, format = "fasta")

# Define colors
color_map <- c(
    "estimated_true" = "#1a3a5c",
    "not_estimated_true" = "#5b9aa0",
    "estimated_not_true" = "#e8b923"
)

# Create haplotype object
h <- haplotype(seqs)
n_haps <- length(labels(h))

cat("Number of unique haplotypes:", n_haps, "\n")

# Map categories and frequencies to haplotypes
hap_indices <- attr(h, "index")
hap_categories <- sapply(seq_len(n_haps), function(i) {
    first_seq_idx <- hap_indices[[i]][1]
    categories[first_seq_idx]
})

hap_freqs <- sapply(seq_len(n_haps), function(i) {
    first_seq_idx <- hap_indices[[i]][1]
    combined_freqs[first_seq_idx]
})

hap_true_csv_indices <- sapply(seq_len(n_haps), function(i) {
    first_seq_idx <- hap_indices[[i]][1]
    true_seq_csv_index[first_seq_idx]
})

node_colors <- color_map[hap_categories]

# Choose a high-contrast text color based on node fill color.
get_contrast_text_color <- function(fill_color) {
    rgb_vals <- col2rgb(fill_color) / 255
    # Relative luminance approximation; lower means darker background.
    luminance <- 0.2126 * rgb_vals[1] + 0.7152 * rgb_vals[2] + 0.0722 * rgb_vals[3]
    if (luminance < 0.5) "#ffffff" else "#111111"
}

# Create haplotype network
net <- haploNet(h)

# Get MST edges
mst_edges <- as.matrix(net)
n_mst_edges <- nrow(mst_edges)

# Get alternative links
alt_links <- attr(net, "alter.links")
non_mst_edges <- NULL
n_non_mst_edges <- 0

if (!is.null(alt_links) && length(alt_links) > 0 && is.matrix(alt_links)) {
    non_mst_edges <- alt_links
    n_non_mst_edges <- nrow(alt_links)
}

cat("\nEdge summary:\n")
cat("  MST edges:", n_mst_edges, "\n")
cat("  Non-MST edges:", n_non_mst_edges, "\n")

# Filter alternative edges - keep top N edges with lowest mutation counts
filtered_non_mst <- NULL
n_filtered_non_mst <- 0

if (n_non_mst_edges > 0 && num_alt_edges > 0) {
    # Sort by mutation count (column 3) and take top N
    sorted_idx <- order(non_mst_edges[, 3])
    n_to_keep <- min(num_alt_edges, n_non_mst_edges)
    keep_idx <- sorted_idx[1:n_to_keep]
    filtered_non_mst <- non_mst_edges[keep_idx, , drop = FALSE]
    n_filtered_non_mst <- nrow(filtered_non_mst)
    max_filtered_mutations <- max(filtered_non_mst[, 3])
    cat("  Filtered non-MST edges (top", num_alt_edges, "by lowest mutations):", n_filtered_non_mst, "\n")
    cat("  Max mutations in filtered set:", max_filtered_mutations, "\n")
} else {
    cat("  Filtered non-MST edges: 0 (num_alt_edges =", num_alt_edges, ")\n")
}

# Scale circle sizes
min_radius <- 0.022
max_radius <- 0.075
scaled_freqs <- sqrt(hap_freqs)
if (max(scaled_freqs) > min(scaled_freqs)) {
    scaled_freqs <- (scaled_freqs - min(scaled_freqs)) / (max(scaled_freqs) - min(scaled_freqs))
} else {
    scaled_freqs <- rep(0.5, length(scaled_freqs))
}
circle_radii <- min_radius + scaled_freqs * (max_radius - min_radius)

# Get coordinates
cat("\nCalculating network layout...\n")
pdf(NULL)
coord <- plot(net, show.mutation = FALSE, labels = FALSE)
dev.off()

xy <- data.frame(x = coord$x, y = coord$y)

# Compress layout
center_x <- mean(xy$x)
center_y <- mean(xy$y)
xy$x <- xy$x - center_x
xy$y <- xy$y - center_y

distances <- sqrt(xy$x^2 + xy$y^2)
max_dist <- max(distances)
if (max_dist > 0) {
    compressed_distances <- log1p(distances) / log1p(max_dist) * 0.45
    angles <- atan2(xy$y, xy$x)
    xy$x <- compressed_distances * cos(angles)
    xy$y <- compressed_distances * sin(angles)
}

# Force-directed layout
spread_nodes <- function(xy, radii, iterations = 200, repulsion = 0.03) {
    n <- nrow(xy)
    for (iter in seq_len(iterations)) {
        forces_x <- rep(0, n)
        forces_y <- rep(0, n)
        for (i in seq_len(n - 1)) {
            for (j in (i + 1):n) {
                dx <- xy$x[j] - xy$x[i]
                dy <- xy$y[j] - xy$y[i]
                dist <- sqrt(dx^2 + dy^2)
                min_dist <- (radii[i] + radii[j]) * 2.0
                if (dist < min_dist && dist > 0.001) {
                    force <- repulsion * (min_dist - dist) / dist
                    forces_x[i] <- forces_x[i] - dx * force
                    forces_y[i] <- forces_y[i] - dy * force
                    forces_x[j] <- forces_x[j] + dx * force
                    forces_y[j] <- forces_y[j] + dy * force
                }
            }
        }
        xy$x <- xy$x + forces_x
        xy$y <- xy$y + forces_y
    }
    return(xy)
}

xy <- spread_nodes(xy, circle_radii, iterations = 250, repulsion = 0.035)

# Normalize
x_range <- range(xy$x)
y_range <- range(xy$y)
xy$x <- (xy$x - x_range[1]) / (max(x_range[2] - x_range[1], 0.001))
xy$y <- (xy$y - y_range[1]) / (max(y_range[2] - y_range[1], 0.001))

padding <- 0.1
xy$x <- xy$x * (1 - 2 * padding) + padding
xy$y <- xy$y * (1 - 2 * padding) + padding

# Helper function to clip edge at circle boundaries
# Returns the start and end points of the edge that don't overlap with circles
clip_edge_to_circles <- function(x1, y1, x2, y2, r1, r2) {
    # Calculate direction vector
    dx <- x2 - x1
    dy <- y2 - y1
    edge_len <- sqrt(dx^2 + dy^2)

    if (edge_len < 0.001) {
        return(list(x1 = x1, y1 = y1, x2 = x2, y2 = y2))
    }

    # Normalize direction
    ux <- dx / edge_len
    uy <- dy / edge_len

    # Add small padding to the radii so edges don't touch circles
    pad <- 0.003

    # New start point: move from center of circle 1 along direction by radius
    new_x1 <- x1 + ux * (r1 + pad)
    new_y1 <- y1 + uy * (r1 + pad)

    # New end point: move from center of circle 2 back along direction by radius
    new_x2 <- x2 - ux * (r2 + pad)
    new_y2 <- y2 - uy * (r2 + pad)

    return(list(x1 = new_x1, y1 = new_y1, x2 = new_x2, y2 = new_y2))
}

# Helper function for label position
find_label_position <- function(x1, y1, x2, y2, r1, r2) {
    mid_x <- (x1 + x2) / 2
    mid_y <- (y1 + y2) / 2
    dist_to_from <- sqrt((mid_x - x1)^2 + (mid_y - y1)^2)
    dist_to_to <- sqrt((mid_x - x2)^2 + (mid_y - y2)^2)

    if (dist_to_from < r1 * 1.2 || dist_to_to < r2 * 1.2) {
        edge_len <- sqrt((x2 - x1)^2 + (y2 - y1)^2)
        if (edge_len > 0) {
            best_t <- 0.5
            best_margin <- -Inf
            for (t in seq(0.15, 0.85, by = 0.05)) {
                test_x <- x1 + t * (x2 - x1)
                test_y <- y1 + t * (y2 - y1)
                d1 <- sqrt((test_x - x1)^2 + (test_y - y1)^2) - r1
                d2 <- sqrt((test_x - x2)^2 + (test_y - y2)^2) - r2
                margin <- min(d1, d2)
                if (margin > best_margin) {
                    best_margin <- margin
                    best_t <- t
                }
            }
            mid_x <- x1 + best_t * (x2 - x1)
            mid_y <- y1 + best_t * (y2 - y1)
        }
    }
    return(c(mid_x, mid_y))
}

# Function to draw the network
draw_network <- function(output_file, title_suffix, show_non_mst, non_mst_to_draw = NULL) {
    cat("Generating:", output_file, "\n")
    png(output_file, width = 1100, height = 950, res = 120, bg = "white")

    par(mar = c(3, 1, 4, 9), xpd = NA)
    plot(NULL, xlim = c(0, 1), ylim = c(0, 1),
         xlab = "", ylab = "", xaxt = "n", yaxt = "n",
         bty = "n", asp = 1)

    # Draw non-MST edges first (behind) - clipped to circle boundaries
    if (show_non_mst && !is.null(non_mst_to_draw) && nrow(non_mst_to_draw) > 0) {
        for (i in seq_len(nrow(non_mst_to_draw))) {
            from_idx <- non_mst_to_draw[i, 1]
            to_idx <- non_mst_to_draw[i, 2]
            x1 <- xy$x[from_idx]
            y1 <- xy$y[from_idx]
            x2 <- xy$x[to_idx]
            y2 <- xy$y[to_idx]
            r1 <- circle_radii[from_idx]
            r2 <- circle_radii[to_idx]

            # Clip edge to circle boundaries
            clipped <- clip_edge_to_circles(x1, y1, x2, y2, r1, r2)
            lines(c(clipped$x1, clipped$x2), c(clipped$y1, clipped$y2),
                  col = "#aaaaaa", lwd = 1, lty = 2)
        }
    }

    # Draw MST edges - clipped to circle boundaries
    for (i in seq_len(n_mst_edges)) {
        from_idx <- mst_edges[i, 1]
        to_idx <- mst_edges[i, 2]
        x1 <- xy$x[from_idx]
        y1 <- xy$y[from_idx]
        x2 <- xy$x[to_idx]
        y2 <- xy$y[to_idx]
        r1 <- circle_radii[from_idx]
        r2 <- circle_radii[to_idx]

        # Clip edge to circle boundaries
        clipped <- clip_edge_to_circles(x1, y1, x2, y2, r1, r2)
        lines(c(clipped$x1, clipped$x2), c(clipped$y1, clipped$y2),
              col = "#2c2c2c", lwd = 3)
    }

    # Draw nodes
    for (i in seq_len(n_haps)) {
        symbols(xy$x[i], xy$y[i], circles = circle_radii[i],
                inches = FALSE, add = TRUE,
                bg = node_colors[i], fg = "#333333", lwd = 2)
    }

    # Draw node labels for true haplotypes: original row index in true CSV.
    for (i in seq_len(n_haps)) {
        if (hap_categories[i] != "estimated_not_true" && !is.na(hap_true_csv_indices[i])) {
            label_color <- get_contrast_text_color(node_colors[i])
            text(xy$x[i], xy$y[i], labels = as.character(hap_true_csv_indices[i]),
                 cex = 0.72, col = label_color, font = 2)
        }
    }

    # Draw MST mutation labels
    for (i in seq_len(n_mst_edges)) {
        from_idx <- mst_edges[i, 1]
        to_idx <- mst_edges[i, 2]
        n_mutations <- mst_edges[i, 3]

        x1 <- xy$x[from_idx]
        y1 <- xy$y[from_idx]
        x2 <- xy$x[to_idx]
        y2 <- xy$y[to_idx]
        r1 <- circle_radii[from_idx]
        r2 <- circle_radii[to_idx]

        pos <- find_label_position(x1, y1, x2, y2, r1, r2)

        label_text <- as.character(n_mutations)
        tw <- strwidth(label_text, cex = 0.65) * 0.6 + 0.008
        th <- strheight(label_text, cex = 0.65) * 0.6 + 0.005
        rect(pos[1] - tw, pos[2] - th, pos[1] + tw, pos[2] + th,
             col = "white", border = "#aaaaaa", lwd = 0.5)
        text(pos[1], pos[2], label_text, cex = 0.65, col = "#111111", font = 2)
    }

    # Draw non-MST mutation labels (smaller)
    if (show_non_mst && !is.null(non_mst_to_draw) && nrow(non_mst_to_draw) > 0) {
        for (i in seq_len(nrow(non_mst_to_draw))) {
            from_idx <- non_mst_to_draw[i, 1]
            to_idx <- non_mst_to_draw[i, 2]
            n_mutations <- non_mst_to_draw[i, 3]

            x1 <- xy$x[from_idx]
            y1 <- xy$y[from_idx]
            x2 <- xy$x[to_idx]
            y2 <- xy$y[to_idx]
            r1 <- circle_radii[from_idx]
            r2 <- circle_radii[to_idx]

            pos <- find_label_position(x1, y1, x2, y2, r1, r2)

            label_text <- as.character(n_mutations)
            tw <- strwidth(label_text, cex = 0.5) * 0.6 + 0.005
            th <- strheight(label_text, cex = 0.5) * 0.6 + 0.003
            rect(pos[1] - tw, pos[2] - th, pos[1] + tw, pos[2] + th,
                 col = "white", border = "#cccccc", lwd = 0.3)
            text(pos[1], pos[2], label_text, cex = 0.5, col = "#666666", font = 1)
        }
    }

    # Legend text based on mode
    if (show_non_mst) {
        legend_text <- c("Estimated True Haplotype",
                        "Not Estimated True Haplotype",
                        "Estimated Not True Haplotype",
                        "",
                        "MST edge (thick solid)",
                        "Other edge (thin dotted)",
                        "",
                        "Node label (true only) = true CSV index",
                        "",
                        "Circle size = frequency",
                        "Numbers = mutations")
        legend_pch <- c(21, 21, 21, NA, NA, NA, NA, NA, NA, NA)
        legend_pt_bg <- c("#1a3a5c", "#5b9aa0", "#e8b923", NA, NA, NA, NA, NA, NA, NA)
        legend_lty <- c(NA, NA, NA, NA, 1, 2, NA, NA, NA, NA)
        legend_lwd <- c(NA, NA, NA, NA, 3, 1, NA, NA, NA, NA)
        legend_col <- c("black", "black", "black", NA, "#2c2c2c", "#aaaaaa", NA, NA, NA, NA)
    } else {
        legend_text <- c("Estimated True Haplotype",
                        "Not Estimated True Haplotype",
                        "Estimated Not True Haplotype",
                        "",
                        "MST edge",
                        "",
                        "Node label (true only) = true CSV index",
                        "",
                        "Circle size = frequency",
                        "Numbers = mutations")
        legend_pch <- c(21, 21, 21, NA, NA, NA, NA, NA, NA, NA)
        legend_pt_bg <- c("#1a3a5c", "#5b9aa0", "#e8b923", NA, NA, NA, NA, NA, NA, NA)
        legend_lty <- c(NA, NA, NA, NA, 1, NA, NA, NA, NA, NA)
        legend_lwd <- c(NA, NA, NA, NA, 3, NA, NA, NA, NA, NA)
        legend_col <- c("black", "black", "black", NA, "#2c2c2c", NA, NA, NA, NA, NA)
    }

    legend("right",
           inset = c(-0.28, 0),
           legend = legend_text,
           pch = legend_pch,
           pt.bg = legend_pt_bg,
           pt.cex = 2,
           lty = legend_lty,
           lwd = legend_lwd,
           col = legend_col,
           title = "Legend",
           cex = 0.7,
           bg = "white",
           box.lwd = 1,
           xpd = TRUE)

    title(main = paste("Haplotype Network:", title_suffix),
          cex.main = 1.1, line = 1.5)

    mtext(paste0("Dark blue: ", sum(hap_categories == "estimated_true"),
                 "  |  Light blue: ", sum(hap_categories == "not_estimated_true"),
                 "  |  Yellow: ", sum(hap_categories == "estimated_not_true")),
          side = 1, line = 1.5, cex = 0.8)

    dev.off()
}

# Generate all three versions
cat("\n========================================\n")
cat("Generating three versions of the network\n")
cat("========================================\n\n")

# 1. MST only (clean)
draw_network(
    paste0(output_prefix, "_mst_only.png"),
    "MST Only (Clean Tree)",
    show_non_mst = FALSE,
    non_mst_to_draw = NULL
)

# 2. Filtered (MST + close alternatives)
draw_network(
    paste0(output_prefix, "_filtered.png"),
    paste0("MST + Filtered Alternatives (", n_filtered_non_mst, " edges)"),
    show_non_mst = TRUE,
    non_mst_to_draw = filtered_non_mst
)

# 3. All edges
draw_network(
    paste0(output_prefix, "_all.png"),
    paste0("All Edges (", n_non_mst_edges, " alternative edges)"),
    show_non_mst = TRUE,
    non_mst_to_draw = non_mst_edges
)

# Cleanup
unlink(temp_fasta)

cat("\n========================================\n")
cat("Done! Generated three files:\n")
cat("  1.", paste0(output_prefix, "_mst_only.png"), "- Clean MST tree\n")
cat("  2.", paste0(output_prefix, "_filtered.png"), "- MST + close alternatives\n")
cat("  3.", paste0(output_prefix, "_all.png"), "- All edges\n")
cat("========================================\n")
