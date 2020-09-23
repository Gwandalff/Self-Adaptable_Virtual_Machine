/**
 */
package miniJava;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Load Image</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link miniJava.LoadImage#getPath <em>Path</em>}</li>
 * </ul>
 *
 * @see miniJava.MiniJavaPackage#getLoadImage()
 * @model
 * @generated
 */
public interface LoadImage extends Expression {
	/**
	 * Returns the value of the '<em><b>Path</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Path</em>' containment reference.
	 * @see #setPath(Expression)
	 * @see miniJava.MiniJavaPackage#getLoadImage_Path()
	 * @model containment="true"
	 * @generated
	 */
	Expression getPath();

	/**
	 * Sets the value of the '{@link miniJava.LoadImage#getPath <em>Path</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Path</em>' containment reference.
	 * @see #getPath()
	 * @generated
	 */
	void setPath(Expression value);

} // LoadImage
