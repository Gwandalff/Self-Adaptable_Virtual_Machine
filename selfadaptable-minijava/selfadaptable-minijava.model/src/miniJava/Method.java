/**
 */
package miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.EMap;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Method</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link miniJava.Method#isIsabstract <em>Isabstract</em>}</li>
 *   <li>{@link miniJava.Method#isIsstatic <em>Isstatic</em>}</li>
 *   <li>{@link miniJava.Method#getParams <em>Params</em>}</li>
 *   <li>{@link miniJava.Method#getBody <em>Body</em>}</li>
 *   <li>{@link miniJava.Method#getCache <em>Cache</em>}</li>
 * </ul>
 *
 * @see miniJava.MiniJavaPackage#getMethod()
 * @model
 * @generated
 */
public interface Method extends Member {
	/**
	 * Returns the value of the '<em><b>Isabstract</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Isabstract</em>' attribute.
	 * @see #setIsabstract(boolean)
	 * @see miniJava.MiniJavaPackage#getMethod_Isabstract()
	 * @model
	 * @generated
	 */
	boolean isIsabstract();

	/**
	 * Sets the value of the '{@link miniJava.Method#isIsabstract <em>Isabstract</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Isabstract</em>' attribute.
	 * @see #isIsabstract()
	 * @generated
	 */
	void setIsabstract(boolean value);

	/**
	 * Returns the value of the '<em><b>Isstatic</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Isstatic</em>' attribute.
	 * @see #setIsstatic(boolean)
	 * @see miniJava.MiniJavaPackage#getMethod_Isstatic()
	 * @model
	 * @generated
	 */
	boolean isIsstatic();

	/**
	 * Sets the value of the '{@link miniJava.Method#isIsstatic <em>Isstatic</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Isstatic</em>' attribute.
	 * @see #isIsstatic()
	 * @generated
	 */
	void setIsstatic(boolean value);

	/**
	 * Returns the value of the '<em><b>Params</b></em>' containment reference list.
	 * The list contents are of type {@link miniJava.Parameter}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Params</em>' containment reference list.
	 * @see miniJava.MiniJavaPackage#getMethod_Params()
	 * @model containment="true"
	 * @generated
	 */
	EList<Parameter> getParams();

	/**
	 * Returns the value of the '<em><b>Body</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Body</em>' containment reference.
	 * @see #setBody(Block)
	 * @see miniJava.MiniJavaPackage#getMethod_Body()
	 * @model containment="true"
	 * @generated
	 */
	Block getBody();

	/**
	 * Sets the value of the '{@link miniJava.Method#getBody <em>Body</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Body</em>' containment reference.
	 * @see #getBody()
	 * @generated
	 */
	void setBody(Block value);

	/**
	 * Returns the value of the '<em><b>Cache</b></em>' map.
	 * The key is of type {@link miniJava.Clazz},
	 * and the value is of type {@link miniJava.Method},
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Cache</em>' map.
	 * @see miniJava.MiniJavaPackage#getMethod_Cache()
	 * @model mapType="miniJava.ClazzToMethodMap&lt;miniJava.Clazz, miniJava.Method&gt;"
	 * @generated
	 */
	EMap<Clazz, Method> getCache();

} // Method
