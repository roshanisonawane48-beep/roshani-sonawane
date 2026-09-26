import matplotlib.pyplot as plt

# 1. Simple Datasets

# Line Plot: Hours studied vs. Exam score over 5 days
days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
hours_studied = [1, 2, 3, 5, 6]

students = ["Alice", "Bob", "Charlie", "David"]
books_read = [4, 7, 2, 5]

# Histogram: Test scores of 10 students
test_scores = [55, 62, 65, 70, 72, 75, 78, 85, 88, 95]



fig, axes = plt.subplots(
    figsize=(12, 4),
    nrows=1,
    ncols=3
)

fig.suptitle(
    "Student Data Visualization",
    fontsize=14,
    fontweight="bold"
)



axes[0].plot(
    days,
    hours_studied,
    marker="o",
    color="pink",
    linewidth=2
)

axes[0].set_title("Line Plot: Study Hours")
axes[0].set_xlabel("Day of Week")
axes[0].set_ylabel("Hours Studied")
axes[0].grid(True)


# 4. Bar Chart
axes[1].bar(
    students,
    books_read,
    color=["pink", "skyblue", "lightgreen", "orange"]
)

axes[1].set_title("Bar Chart: Books Read")
axes[1].set_xlabel("Students")
axes[1].set_ylabel("Number of Books")


# 5. Histogram
axes[2].hist(
    test_scores,
    bins=[50, 60, 70, 80, 90, 100],
    color="purple",
    edgecolor="brown"
)

axes[2].set_title("Histogram: Score Distribution")
axes[2].set_xlabel("Score Ranges")
axes[2].set_ylabel("Number of Students")


# 6. Adjust spacing
plt.tight_layout()

# 7. Display the plots
plt.show()
