import models
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)
valid_userid = ['00sab00','1234','zippy','zburt5','joshua','dorothy w','rebecca','walker557','samantha','raeanne','kimmie','cassie','moore222']


@app.route('/')
def view():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend_top5():
    # Accept both JSON and form data
    if request.is_json:
        user_name = request.json.get('username')
    else:
        user_name = request.form.get('User Name') or request.form.get('username')
    print('User name=', user_name)

    if user_name in valid_userid and request.method == 'POST':
        top20_products = models.recommend_products(user_name)
        details = models.top5_products(top20_products)
        # Prepare JSON response for frontend
        products = []
        for item in details:
            products.append({
                'name': item['name'],
                'Category': item['category'],
                'Brand': item['brand'],
                'price': item['price'],
                'description': item['description']
            })
        return jsonify(products)
    elif not user_name in valid_userid:
        return jsonify([])
    else:
        return jsonify([])


if __name__ == '__main__':
    app.debug = False
    app.run()