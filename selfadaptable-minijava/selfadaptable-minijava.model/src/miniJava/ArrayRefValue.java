/**
 */
package miniJava;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Array Ref Value</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link miniJava.ArrayRefValue#getInstance <em>Instance</em>}</li>
 * </ul>
 *
 * @see miniJava.MiniJavaPackage#getArrayRefValue()
 * @model annotation="aspect"
 *        annotation="RuntimeData"
 * @generated
 */
public interface ArrayRefValue extends Value {
	/**
	 * Returns the value of the '<em><b>Instance</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Instance</em>' reference.
	 * @see #setInstance(ArrayInstance)
	 * @see miniJava.MiniJavaPackage#getArrayRefValue_Instance()
	 * @model
	 * @generated
	 */
	ArrayInstance getInstance();

	/**
	 * Sets the value of the '{@link miniJava.ArrayRefValue#getInstance <em>Instance</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Instance</em>' reference.
	 * @see #getInstance()
	 * @generated
	 */
	void setInstance(ArrayInstance value);

} // ArrayRefValue
