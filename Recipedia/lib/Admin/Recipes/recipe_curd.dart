import 'package:flutter/material.dart';
import 'package:recipedia/Admin/recipes/view_recipes.dart';

import 'add_recipe.dart';

class RecipeCURD extends StatefulWidget {
  const RecipeCURD({
    super.key,
  });

  @override
  State<RecipeCURD> createState() => _RecipeCURDState();
}

class _RecipeCURDState extends State<RecipeCURD> {
  /*Future<void> getRecipeData() async {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (BuildContext context) {
          dialogContext = context;
          return ProgressBar(
            message: "Loading..",
          );
        },
      );
    });
    int count = await RecipeModel().getRecipeCount();
    print(count);
    int recipeID = 101;

    for (int i = 0; i < count - 1; i++, recipeID++) {
      recipeIDs.add(recipeID.toString());
      recipeDescription.add((await RecipeModel()
          .getRecipeData(recipeID.toString(), 'recipeDescription'))!);
      recipeName.add((await RecipeModel()
          .getRecipeData(recipeID.toString(), 'recipeName'))!);
      recipeURL.add((await RecipeModel()
          .getRecipeData(recipeID.toString(), 'recipeURL'))!);
      recipeRating.add((await RecipeModel()
          .getRecipeData(recipeID.toString(), 'recipeRating'))!);
      recipeTime.add((await RecipeModel()
          .getRecipeData(recipeID.toString(), 'recipeTime'))!);
      recipeIngredients.add((await RecipeModel()
          .getRecipeData(recipeID.toString(), 'recipeIngredients'))!);
      recipes.add(
        Recipes(
            recipeID: recipeIDs[i],
            recipeName: recipeName[i],
            recipeDescription: recipeDescription[i],
            recipeIngredients: [recipeIngredients[i]],
            recipeRating: recipeRating[i],
            recipeTime: recipeTime[i],
            recipeURL: recipeURL[i]),
      );
      print(recipes[i].recipeName +
          recipes[i].recipeDescription +
          recipes[i].recipeURL +
          recipes[i].recipeRating +
          recipes[i].recipeTime +
          recipes[i].recipeIngredients.join(""));
    }
    Navigator.pop(dialogContext!);
  }

  @override
  void initState() {
    getRecipeData();
    super.initState();
  }*/

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
        body: Stack(
      children: [
        Positioned.fill(
          child: Image.asset(
            'assets/curd.jpg',
            fit: BoxFit.cover,
          ),
        ),
        Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              height: size.height * 0.2,
            ),
            //Recipe Title
            Container(
              alignment: Alignment.centerLeft,
              padding: const EdgeInsets.only(
                left: 30,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.end,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: const [
                  Text(
                    'Recipes',
                    style: TextStyle(
                        color: Color(0XFFFF4F5A),
                        fontSize: 38.0,
                        fontWeight: FontWeight.w800),
                  ),
                ],
              ),
            ),
            SizedBox(
              height: size.height * 0.05,
            ),
            Container(
              alignment: Alignment.center,
              width: size.width * 0.95,
              padding: const EdgeInsets.only(left: 30.0, right: 30.0),
              child: Column(
                children: [
                  Container(
                    width: MediaQuery.of(context).size.width,
                    height: 40,
                    margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                    decoration:
                        BoxDecoration(borderRadius: BorderRadius.circular(90)),
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          PageRouteBuilder(
                            transitionDuration: const Duration(seconds: 1),
                            transitionsBuilder:
                                (context, animation, animationTime, child) {
                              animation = CurvedAnimation(
                                  parent: animation,
                                  curve: Curves.fastLinearToSlowEaseIn);
                              return ScaleTransition(
                                scale: animation,
                                alignment: Alignment.center,
                                child: child,
                              );
                            },
                            pageBuilder: (context, animation, animationTime) {
                              return const AddRecipe();
                            },
                          ),
                        );
                      },
                      style: ButtonStyle(
                          backgroundColor:
                              MaterialStateProperty.resolveWith((states) {
                            if (states.contains(MaterialState.pressed)) {
                              return Colors.blueGrey;
                            }
                            return const Color(0XFFFF4F5A);
                          }),
                          shape: MaterialStateProperty.all(
                              RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(30)))),
                      child: const Text(
                        'Add Recipe',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: size.height * 0.02,
                  ),
                  Container(
                    width: MediaQuery.of(context).size.width,
                    height: 40,
                    margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                    decoration:
                        BoxDecoration(borderRadius: BorderRadius.circular(90)),
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          PageRouteBuilder(
                            transitionDuration: const Duration(seconds: 1),
                            transitionsBuilder:
                                (context, animation, animationTime, child) {
                              animation = CurvedAnimation(
                                  parent: animation,
                                  curve: Curves.fastLinearToSlowEaseIn);
                              return ScaleTransition(
                                scale: animation,
                                alignment: Alignment.center,
                                child: child,
                              );
                            },
                            pageBuilder: (context, animation, animationTime) {
                              return const ViewRecipes();
                            },
                          ),
                        );
                      },
                      style: ButtonStyle(
                          backgroundColor:
                              MaterialStateProperty.resolveWith((states) {
                            if (states.contains(MaterialState.pressed)) {
                              return Colors.blueGrey;
                            }
                            return const Color(0XFFFF4F5A);
                          }),
                          shape: MaterialStateProperty.all(
                              RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(30)))),
                      child: const Text(
                        'View Recipes',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    height: size.height * 0.02,
                  ),
                  Container(
                    width: MediaQuery.of(context).size.width,
                    height: 40,
                    margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                    decoration:
                        BoxDecoration(borderRadius: BorderRadius.circular(90)),
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.pop(context);
                      },
                      style: ButtonStyle(
                          backgroundColor:
                              MaterialStateProperty.resolveWith((states) {
                            if (states.contains(MaterialState.pressed)) {
                              return Colors.blueGrey;
                            }
                            return const Color(0XFFFF4F5A);
                          }),
                          shape: MaterialStateProperty.all(
                              RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(30)))),
                      child: const Text(
                        'Back',
                        style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ],
    )
        /*Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            height: size.height * 0.02,
          ),
          
          SizedBox(
            height: size.height * 0.02,
          ),
          //Recipe Title
          Container(
            alignment: Alignment.centerLeft,
            padding: const EdgeInsets.only(
              left: 30,
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.end,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  'Recipes',
                  style: TextStyle(
                      color: Color(0XFFFF4F5A),
                      fontSize: 38.0,
                      fontWeight: FontWeight.w800),
                ),
              ],
            ),
          ),
          SizedBox(
            height: size.height * 0.02,
          ),
          Container(
            alignment: Alignment.center,
            padding: const EdgeInsets.only(left: 30.0, right: 30.0),
            child: Column(
              children: [
                SizedBox(
                  height: size.height * 0.01,
                ),
                Container(
                  width: MediaQuery.of(context).size.width,
                  height: 40,
                  margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                  decoration:
                      BoxDecoration(borderRadius: BorderRadius.circular(90)),
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        PageRouteBuilder(
                          transitionDuration: const Duration(seconds: 1),
                          transitionsBuilder:
                              (context, animation, animationTime, child) {
                            animation = CurvedAnimation(
                                parent: animation,
                                curve: Curves.fastLinearToSlowEaseIn);
                            return ScaleTransition(
                              scale: animation,
                              alignment: Alignment.center,
                              child: child,
                            );
                          },
                          pageBuilder: (context, animation, animationTime) {
                            return AddRecipe();
                          },
                        ),
                      );
                    },
                    style: ButtonStyle(
                        backgroundColor:
                            MaterialStateProperty.resolveWith((states) {
                          if (states.contains(MaterialState.pressed)) {
                            return Colors.blueGrey;
                          }
                          return const Color(0XFFFF4F5A);
                        }),
                        shape: MaterialStateProperty.all(RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30)))),
                    child: const Text(
                      'Add Recipe',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                  ),
                ),
                SizedBox(
                  height: size.height * 0.01,
                ),
                Container(
                  width: MediaQuery.of(context).size.width,
                  height: 40,
                  margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                  decoration:
                      BoxDecoration(borderRadius: BorderRadius.circular(90)),
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        PageRouteBuilder(
                          transitionDuration: const Duration(seconds: 1),
                          transitionsBuilder:
                              (context, animation, animationTime, child) {
                            animation = CurvedAnimation(
                                parent: animation,
                                curve: Curves.fastLinearToSlowEaseIn);
                            return ScaleTransition(
                              scale: animation,
                              alignment: Alignment.center,
                              child: child,
                            );
                          },
                          pageBuilder: (context, animation, animationTime) {
                            return const ViewRecipes();
                          },
                        ),
                      );
                    },
                    style: ButtonStyle(
                        backgroundColor:
                            MaterialStateProperty.resolveWith((states) {
                          if (states.contains(MaterialState.pressed)) {
                            return Colors.blueGrey;
                          }
                          return const Color(0XFFFF4F5A);
                        }),
                        shape: MaterialStateProperty.all(RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30)))),
                    child: const Text(
                      'View Recipes',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                  ),
                ),
                SizedBox(
                  height: size.height * 0.01,
                ),
                Container(
                  width: MediaQuery.of(context).size.width,
                  height: 40,
                  margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                  decoration:
                      BoxDecoration(borderRadius: BorderRadius.circular(90)),
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    style: ButtonStyle(
                        backgroundColor:
                            MaterialStateProperty.resolveWith((states) {
                          if (states.contains(MaterialState.pressed)) {
                            return Colors.blueGrey;
                          }
                          return const Color(0XFFFF4F5A);
                        }),
                        shape: MaterialStateProperty.all(RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30)))),
                    child: const Text(
                      'Back',
                      style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 16),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),*/
        );
  }
}
