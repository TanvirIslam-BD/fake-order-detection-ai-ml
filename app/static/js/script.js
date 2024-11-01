// static/js/script.js
document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector('form');
    const hyperparamsInput = document.querySelector('textarea[name="hyperparameters"]');

    // Example: Provide default hyperparameters for RandomForest
    document.querySelector('select[name="model"]').addEventListener('change', function () {
        // if (this.value === 'RandomForest') {
        //     hyperparamsInput.value = '{"n_estimators": 100, "max_depth": null}';
        // } else {
        //     hyperparamsInput.value = '';
        // }
    });
});
