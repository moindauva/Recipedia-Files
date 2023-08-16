import 'package:dynamic_height_grid_view/dynamic_height_grid_view.dart';
import 'package:flutter/material.dart';
import 'Admin/Recipes/recipe_card.dart';
import 'Database/databaseHelper.dart';

class ScanFeature extends StatefulWidget {
  final List itemsScanned;
  final bool validIngredient;

  const ScanFeature({Key? key, required this.itemsScanned, required this.validIngredient,}) : super(key: key);

  @override
  State<ScanFeature> createState() => _ScanFeatureState();
}

class _ScanFeatureState extends State<ScanFeature> {
  Future<List<Map<String, dynamic>>> getRecipesDataFromLocalDatabase() async {
    List<Map<String, dynamic>> recipes =
        await DatabaseHelper().getRecipeByIngredient(widget.itemsScanned);
    //await Future.delayed(const Duration(seconds: 3));
    return recipes;
  }



  /*@override
  void initState() {
    if (widget.itemsScanned==null||widget.itemsScanned.isEmpty) {
      validIngredient == false;
    } else {
      validIngredient == true;
    }
    super.initState();
  }*/

  Widget gridViewForIngredients(String item) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 20,
        vertical: 10,
      ),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        color: const Color(0XFFFF4353),
      ),
      child: Text(item,
          style: const TextStyle(
              fontSize: 16.0,
              fontWeight: FontWeight.bold,
              color: Colors.black54)),
    );
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
              bottomRight: Radius.circular(25),
              bottomLeft: Radius.circular(25)),
        ),
        centerTitle: true,
        title: const Text(
          'Recipes',
          textAlign: TextAlign.center,
        ),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
      ),
      body: SingleChildScrollView(
          child: Column(
        children: [
          SizedBox(
            height: size.height * 0.025,
          ),
          Container(
            alignment: Alignment.centerLeft,
            padding: const EdgeInsets.only(
              left: 25.0,
              right: 25.0,
            ),
            child: const Text(
              'Ingredients Scanned',
              style: TextStyle(
                fontSize: 24.0,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          SizedBox(
            height: size.height * 0.015,
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: widget.validIngredient
                ? Wrap(
                    spacing: 8.0, // spacing between adjacent items
                    runSpacing: 8.0, // spacing between lines
                    children: widget.itemsScanned
                        .map((item) => gridViewForIngredients(item))
                        .toList(),
                  )
                : const Text(
                    'Please Scan the valid ingredient.',
                    style: TextStyle(fontSize: 16.0),
                  ),
          ),
          SizedBox(
            height: size.height * 0.015,
          ),
          widget.validIngredient
              ? FutureBuilder<List<Map<String, dynamic>>>(
                  future: getRecipesDataFromLocalDatabase(),
                  builder: (BuildContext context,
                      AsyncSnapshot<List<Map<String, dynamic>>> snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return SizedBox(
                        height: size.height * 0.8,
                        child: const Center(
                          child: CircularProgressIndicator(),
                        ),
                      );
                    } else if (snapshot.hasError) {
                      return Text('Error: ${snapshot.error}');
                    } else {
                      List<Map<String, dynamic>> recipes = snapshot.data!;
                      return SizedBox(
                        height: size.height * 0.75,
                        child: CustomScrollView(
                          slivers: [
                            SliverDynamicHeightGridView(
                              itemCount: recipes.length,
                              crossAxisCount: 2,
                              builder: (ctx, index) {
                                return RecipeCard(
                                  isAdmin: false,
                                  id: (recipes[index]['recipeId']).toString(),
                                  title: recipes[index]['recipeName'],
                                  description: recipes[index]
                                      ['recipeDescription'],
                                  course: recipes[index]['recipeCourse'],
                                  diet: recipes[index]['recipeDiet'],
                                  cookTime: recipes[index]['recipeTime'],
                                  rating: recipes[index]['recipeRating'],
                                  thumbnailUrl: recipes[index]['recipeURL'],
                                  ingredients: recipes[index]
                                      ['recipeIngredients'],
                                  category: recipes[index]['recipeCategory'],
                                );
                              },
                            ),
                          ],
                        ),
                      );
                    }
                  },
                )
              : SizedBox(
                  height: size.height * 0.8,
                  child: const Center(
                    child: Text(
                      'Recipes Not Found',
                      style: TextStyle(fontSize: 16.0),
                    ),
                  ),
                ),
        ],
      )),
    );
  }
}
