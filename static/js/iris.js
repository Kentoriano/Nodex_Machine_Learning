document.addEventListener("DOMContentLoaded", function () {

    console.log("Probabilities:", probabilities);

    if (probabilities) {

        const data = {
            labels: ['Setosa', 'Versicolor', 'Virginica'],
            datasets: [{
                label: 'Probabilidad (%)',
                data: [
                    probabilities["Iris-setosa"],
                    probabilities["Iris-versicolor"],
                    probabilities["Iris-virginica"]
                ],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.6)',
                    'rgba(255, 206, 86, 0.6)',
                    'rgba(255, 99, 132, 0.6)'
                ]
            }]
        };

        new Chart(document.getElementById('irisChart'), {
            type: 'bar',
            data: data,
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
});